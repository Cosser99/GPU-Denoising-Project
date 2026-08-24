#include "filters/FFTFilter.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            std::fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            std::fprintf(stderr, "*** FAILED - ABORTING\n"); \
            std::exit(1); \
        } \
    } while (0)

#define cufftCheckErrors(call) \
    do { \
        cufftResult __err = (call); \
        if (__err != CUFFT_SUCCESS) { \
            std::fprintf(stderr, "Fatal cuFFT error: %d at %s:%d\n", static_cast<int>(__err), __FILE__, __LINE__); \
            std::fprintf(stderr, "*** FAILED - ABORTING\n"); \
            std::exit(1); \
        } \
    } while (0)

namespace
{
    constexpr unsigned int kBlockSize = 16;

    // bit left shift corresponds to multiplication x2
    size_t nextPowerOfTwo(size_t value)
    {
        size_t result = 1;
        while (result < value)
        {
            result <<= 1;
        }
        return result;
    }

    template <typename PixelT>
    __device__ PixelT fftValue(double value, double lowerLimit, double upperLimit)
    {
        if constexpr (std::is_integral_v<PixelT>)
        {
            const double rounded = floor(value + 0.5);
            return static_cast<PixelT>(fmin(fmax(rounded, lowerLimit), upperLimit));
        }
        else
        {
            return static_cast<PixelT>(value);
        }
    }

    __global__ void makeTransferFunction(double* transferFunction, size_t paddedWidth, size_t paddedHeight, double sigma)
    {
        const size_t x = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t y = threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        if (x >= paddedWidth || y >= paddedHeight)
        {
            return;
        }
        const double dx = static_cast<double>(x) - paddedWidth / 2;
        const double dy = static_cast<double>(y) - paddedHeight / 2;
        // H(u, v) = e^(- D^2(u, v) / (2 * sigma^2) )
        transferFunction[y * paddedWidth + x] = __expf(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
    }

    template <typename PixelT>
    __global__ void makePaddedImage(const PixelT* input, cufftComplex* paddedImage,
                                           int width, int height, int nChannels,
                                           size_t paddedWidth, size_t paddedHeight, size_t paddedSize)
    {
        const size_t x = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t y = threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        const int channel = blockIdx.z;
        if (x >= paddedWidth || y >= paddedHeight || channel >= nChannels)
        {
            return;
        }
        // Step 2. make a new image P x Q using replicate padding.
        // original image is located on the top-left corner
        const int sourceX = x < static_cast<size_t>(width) ? static_cast<int>(x) : width - 1;
        const int sourceY = y < static_cast<size_t>(height) ? static_cast<int>(y) : height - 1;
        const double value = static_cast<double>(input[(static_cast<size_t>(sourceY) * width + sourceX) * nChannels + channel]);
        // Step 3. multiply the image by (-1)^(x+y)
        paddedImage[static_cast<size_t>(channel) * paddedSize + y * paddedWidth + x] =
            make_cuComplex((x + y) % 2 == 0 ? value : -value, 0.0);
    }

    __global__ void multiplyTransferFunction(cufftComplex* paddedImage, const double* transferFunction,
                                             size_t paddedSize, int nChannels)
    {
        const size_t index = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t total = paddedSize * nChannels;
        if (index < total)
        {
            // Step 6. element wise multiplication H(u,v) x F(u, v) = G(u, v)
            const double weight = transferFunction[index % paddedSize];
            paddedImage[index].x *= weight;
            paddedImage[index].y *= weight;
        }
    }

    template <typename PixelT>
    __global__ void extractFilteredImages(const cufftComplex* paddedImage, PixelT* output,
                                          int width, int height, int nChannels,
                                          size_t paddedWidth, size_t paddedSize, double scale, double lowerLimit, double upperLimit)
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int channel = blockIdx.z;
        if (x >= width || y >= height || channel >= nChannels)
        {
            return;
        }
        // step 8. extract the filtered image from top left corner M x N
        double value = paddedImage[static_cast<size_t>(channel) * paddedSize + static_cast<size_t>(y) * paddedWidth + x].x * scale;
        if ((x + y) % 2 != 0)
        {
            value = -value;
        }
        output[(static_cast<size_t>(y) * width + x) * nChannels + channel] = fftValue<PixelT>(value, lowerLimit, upperLimit);
    }
}

namespace idl
{
    template <typename PixelT>
    FFTFilter<PixelT>::FFTFilter(double sigma) : _sigma(sigma)
    {
        if (_sigma <= 0.0)
        {
            throw std::invalid_argument("sigma must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> FFTFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        // Step 1. starting from an input image M x N retrieve padding sizes P = 2M and Q = 2N
        // actually, since successive-doubling method require a power of 2, P and Q are the closest power of 2 >= 2M and 2N
        const size_t paddedWidth = nextPowerOfTwo(2 * static_cast<size_t>(input.width()));
        const size_t paddedHeight = nextPowerOfTwo(2 * static_cast<size_t>(input.height()));
        const size_t paddedSize = paddedWidth * paddedHeight;

        PixelT* deviceInput;
        PixelT* deviceOutput;
        double* deviceTransferFunction;
        cufftComplex* devicePaddedImage;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceTransferFunction, paddedSize * sizeof(double));
        cudaMalloc(&devicePaddedImage, paddedSize * input.nChannels() * sizeof(cufftComplex));
        cudaCheckErrors("cudaMalloc failure");
        cudaEventCreate(&h2dStart); cudaEventCreate(&h2dEnd);
        cudaEventCreate(&kernelStart); cudaEventCreate(&kernelEnd);
        cudaEventCreate(&d2hStart); cudaEventCreate(&d2hEnd);
        cudaCheckErrors("cudaEventCreate failure");
        cudaEventRecord(h2dStart);
        cudaMemcpy(deviceInput, input.data(), input.vectorSize() * sizeof(PixelT), cudaMemcpyHostToDevice);
        cudaEventRecord(h2dEnd);
        cudaCheckErrors("cudaMemcpy H2D failure");

        const dim3 block(kBlockSize, kBlockSize);
        const dim3 paddedGrid((paddedWidth + block.x - 1) / block.x, (paddedHeight + block.y - 1) / block.y);
        const dim3 imageGrid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);

        // Step 5. symmetric filter transfer function P x Q
        // made just once (it is the same for each channel, no need to recompute it in each loop)
        // Same operation as the CPU loop that fills transferFunction, but computed on the GPU.
        cudaEventRecord(kernelStart);
        makeTransferFunction<<<paddedGrid, block>>>(deviceTransferFunction, paddedWidth, paddedHeight, _sigma);
        cudaCheckErrors("transfer function kernel launch failure");

        int dimensions[2] = {static_cast<int>(paddedWidth), static_cast<int>(paddedHeight)};
        cufftHandle plan;
        cufftCheckErrors(cufftPlanMany(&plan, 2, dimensions, nullptr, 1, static_cast<int>(paddedSize),
                                       nullptr, 1, static_cast<int>(paddedSize), CUFFT_C2C, input.nChannels()));

        // each channel operates independently
        const dim3 channelPaddedGrid(paddedGrid.x, paddedGrid.y, input.nChannels());
        // Step 2. make a new image P x Q using replicate padding.
        // prepare all channels in a single 3D grid.
        makePaddedImage<<<channelPaddedGrid, block>>>(deviceInput, devicePaddedImage, input.width(), input.height(),
                                                             input.nChannels(), paddedWidth, paddedHeight, paddedSize);
        cudaCheckErrors("padding kernel launch failure");

        // Step 4. compute the DFT
        // cuFFT computes all channels with one batched plan.
        cufftCheckErrors(cufftExecC2C(plan, devicePaddedImage, devicePaddedImage, CUFFT_FORWARD));

        const dim3 multiplyBlock(256);
        const dim3 multiplyGrid((paddedSize * input.nChannels() + multiplyBlock.x - 1) / multiplyBlock.x);
        // Step 6. element wise multiplication H(u,v) x F(u, v) = G(u, v)
        multiplyTransferFunction<<<multiplyGrid, multiplyBlock>>>(devicePaddedImage, deviceTransferFunction,
                                                                   paddedSize, input.nChannels());
        cudaCheckErrors("transfer multiplication kernel launch failure");

        // step 7. filtered image is IDFT(G)
        cufftCheckErrors(cufftExecC2C(plan, devicePaddedImage, devicePaddedImage, CUFFT_INVERSE));

        const dim3 channelImageGrid(imageGrid.x, imageGrid.y, input.nChannels());
        // step 8. extract the filtered image from top left corner M x N
        // extract all channels in a single 3D grid.
        extractFilteredImages<<<channelImageGrid, block>>>(devicePaddedImage, deviceOutput, input.width(), input.height(),
                                                           input.nChannels(), paddedWidth, paddedSize,
                                                           1.0 / static_cast<double>(paddedSize),
                                                           static_cast<double>(std::numeric_limits<PixelT>::lowest()),
                                                           static_cast<double>(std::numeric_limits<PixelT>::max()));
        cudaCheckErrors("extraction kernel launch failure");

        cudaEventRecord(kernelEnd);
        cudaEventRecord(d2hStart);
        cudaMemcpy(output.data(), deviceOutput, output.vectorSize() * sizeof(PixelT), cudaMemcpyDeviceToHost);
        cudaEventRecord(d2hEnd);
        cudaEventSynchronize(d2hEnd);
        cudaCheckErrors("cudaMemcpy D2H failure");
        float h2dMs, kernelMs, d2hMs, deviceTotalMs;
        cudaEventElapsedTime(&h2dMs, h2dStart, h2dEnd); cudaEventElapsedTime(&kernelMs, kernelStart, kernelEnd);
        cudaEventElapsedTime(&d2hMs, d2hStart, d2hEnd); cudaEventElapsedTime(&deviceTotalMs, h2dStart, d2hEnd);
        cudaCheckErrors("cudaEventElapsedTime failure");
        this->setFilterTiming({h2dMs, kernelMs, d2hMs, deviceTotalMs});
        cudaEventDestroy(h2dStart); cudaEventDestroy(h2dEnd); cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd); cudaEventDestroy(d2hStart); cudaEventDestroy(d2hEnd);
        cudaCheckErrors("cudaEventDestroy failure");
        cufftCheckErrors(cufftDestroy(plan));
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceTransferFunction);
        cudaFree(devicePaddedImage);
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        return output;
    }

    template <typename PixelT> std::string FFTFilter<PixelT>::name() const { return "FFT Gaussian Lowpass Filter"; }
    template <typename PixelT> Architecture FFTFilter<PixelT>::arch() const { return Architecture::GPU; }
    template class FFTFilter<uint8_t>;
    template class FFTFilter<float>;
}
