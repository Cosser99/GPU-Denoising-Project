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

    __global__ void makeTransferFunction(double* transferFunction, size_t paddedWidth, size_t paddedHeight, double sigma)
    {
        const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t idy = threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        
        if (idx >= paddedWidth || idy >= paddedHeight)
        {
            return;
        }

        const double dx = static_cast<double>(idx) - paddedWidth / 2;
        const double dy = static_cast<double>(idy) - paddedHeight / 2;
        // H(u, v) = e^(- D^2(u, v) / (2 * sigma^2) )
        transferFunction[idy * paddedWidth + idx] = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
    }

    template <typename PixelT>
    __global__ void makePaddedImage(const PixelT* input, cufftDoubleComplex* paddedImage,
                                    int width, int height, int nChannels, int channel,
                                    size_t paddedWidth, size_t paddedHeight)
    {
        const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        const size_t idy = threadIdx.y + static_cast<size_t>(blockIdx.y) * blockDim.y;
        
        if (idx >= paddedWidth || idy >= paddedHeight)
        {
            return;
        }

        // Step 2. make a new image P x Q using replicate padding.
        // original image is located on the top-left corner
        const int xx = idx < static_cast<size_t>(width) ? static_cast<int>(idx) : width - 1;
        const int yy = idy < static_cast<size_t>(height) ? static_cast<int>(idy) : height - 1;
        const double value = static_cast<double>(input[(static_cast<size_t>(yy) * width + xx) * nChannels + channel]);
        // Step 3. multiply the image by (-1)^(x+y)
        paddedImage[idy * paddedWidth + idx] = make_cuDoubleComplex((idx + idy) % 2 == 0 ? value : -value, 0.0);
    }

    __global__ void multiplyTransferFunction(cufftDoubleComplex* paddedImage, const double* transferFunction, size_t count)
    {
        const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
        
        if (idx < count)
        {
            // Step 6. element wise multiplication H(u,v) x F(u, v) = G(u, v)
            paddedImage[idx].x *= transferFunction[idx];
            paddedImage[idx].y *= transferFunction[idx];
        }
    }

    template <typename PixelT>
    __global__ void extractFilteredImage(const cufftDoubleComplex* paddedImage, PixelT* output,
                                         int width, int height, int nChannels, int channel,
                                         size_t paddedWidth, double scale, double lowerLimit, double upperLimit)
    {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int idy = threadIdx.y + blockIdx.y * blockDim.y;
        if (idx >= width || idy >= height)
        {
            return;
        }
        // step 8. extract the filtered image from top left corner M x N
        double value = paddedImage[static_cast<size_t>(idy) * paddedWidth + idx].x * scale;
        if ((idx + idy) % 2 != 0)
        {
            value = -value;
        }
        
        if constexpr (std::is_integral_v<PixelT>)
        {
            // working with frequencies may lead to negative numbers, so to be sure to have no errors we
            // clamp the result between min and max (for uin8t 0 255) 
            output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] =
                static_cast<PixelT>(fmin(fmax(floor(value + 0.5), lowerLimit), upperLimit));
        }
        else
        {
            output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = static_cast<PixelT>(value);
        }
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
        cufftDoubleComplex* devicePaddedImage;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;
        
        // allocate device memory
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceTransferFunction, paddedSize * sizeof(double));
        cudaMalloc(&devicePaddedImage, paddedSize * sizeof(cufftDoubleComplex));
        cudaCheckErrors("cudaMalloc failure");
        
        // cuda events are used to take the time needed for each operation
        // saving the time allows us to compare the non optimized with the optimized version 
        cudaEventCreate(&h2dStart); 
        cudaEventCreate(&h2dEnd);
        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelEnd);
        cudaEventCreate(&d2hStart);
        cudaEventCreate(&d2hEnd);
        cudaCheckErrors("cudaEventCreate failure");
        
        // copy data from host to device
        cudaEventRecord(h2dStart);
        cudaMemcpy(deviceInput, input.data(), input.vectorSize() * sizeof(PixelT), cudaMemcpyHostToDevice);
        cudaEventRecord(h2dEnd);
        cudaCheckErrors("cudaMemcpy H2D failure");

        // launch kernel(s) since to use cufft we must split the execution 
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 paddedGrid((paddedWidth + block.x - 1) / block.x, (paddedHeight + block.y - 1) / block.y);
        const dim3 imageGrid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);

        // Step 5. symmetric filter transfer function P x Q
        // made just once (it is the same for each channel, no need to recompute it in each loop)
        // Same operation as the CPU loop that fills transferFunction, but computed on the GPU.
        cudaEventRecord(kernelStart);
        makeTransferFunction<<<paddedGrid, block>>>(deviceTransferFunction, paddedWidth, paddedHeight, _sigma);
        cudaDeviceSynchronize();cudaCheckErrors("transfer function kernel launch failure");

        cufftHandle plan;
        cufftCheckErrors(cufftPlan2d(&plan, static_cast<int>(paddedHeight), static_cast<int>(paddedWidth), CUFFT_Z2Z));
        const dim3 multiplyBlock(256);
        const dim3 multiplyGrid((paddedSize + multiplyBlock.x - 1) / multiplyBlock.x);
        const double inverseScale = 1.0 / static_cast<double>(paddedSize);

        // each channel operates independently
        for (int channel = 0; channel < input.nChannels(); channel++)
        {
            // step 2. making padded image
            makePaddedImage<<<paddedGrid, block>>>(deviceInput, devicePaddedImage, input.width(), input.height(),
                                                   input.nChannels(), channel, paddedWidth, paddedHeight);
            cudaDeviceSynchronize();cudaCheckErrors("padding kernel launch failure");

            // Step 4. compute the DFT
            cufftCheckErrors(cufftExecZ2Z(plan, devicePaddedImage, devicePaddedImage, CUFFT_FORWARD));

            // Step 6. element wise multiplication H(u,v) x F(u, v) = G(u, v)
            multiplyTransferFunction<<<multiplyGrid, multiplyBlock>>>(devicePaddedImage, deviceTransferFunction, paddedSize);
            cudaCheckErrors("transfer multiplication kernel launch failure");

            // step 7. filtered image is IDFT(G)
            cufftCheckErrors(cufftExecZ2Z(plan, devicePaddedImage, devicePaddedImage, CUFFT_INVERSE));

            // step 8. extract the filtered image from top left corner M x N
            extractFilteredImage<<<imageGrid, block>>>(devicePaddedImage, deviceOutput, input.width(), input.height(),
                                                        input.nChannels(), channel, paddedWidth, inverseScale,
                                                        static_cast<double>(std::numeric_limits<PixelT>::lowest()),
                                                        static_cast<double>(std::numeric_limits<PixelT>::max()));
            cudaDeviceSynchronize();cudaCheckErrors("extraction kernel launch failure");
        }

        cudaEventRecord(kernelEnd);

        // copy data from device to host
        cudaEventRecord(d2hStart);
        cudaMemcpy(output.data(), deviceOutput, output.vectorSize() * sizeof(PixelT), cudaMemcpyDeviceToHost);
        cudaEventRecord(d2hEnd);
        cudaEventSynchronize(d2hEnd);
        cudaCheckErrors("cudaMemcpy D2H failure");

        float h2dMs, kernelMs, d2hMs, deviceTotalMs;
        cudaEventElapsedTime(&h2dMs, h2dStart, h2dEnd); cudaEventElapsedTime(&kernelMs, kernelStart, kernelEnd);
        cudaEventElapsedTime(&d2hMs, d2hStart, d2hEnd); cudaEventElapsedTime(&deviceTotalMs, h2dStart, d2hEnd);
        cudaCheckErrors("cudaEventElapsedTime failure");
        
        this->setGpuTiming({true, h2dMs, kernelMs, d2hMs, deviceTotalMs});
        
        // "freeing" events
        cudaEventDestroy(h2dStart);
        cudaEventDestroy(h2dEnd);
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
        cudaEventDestroy(d2hStart);
        cudaEventDestroy(d2hEnd);
        cudaCheckErrors("cudaEventDestroy failure");
        cufftCheckErrors(cufftDestroy(plan));

        // free memory
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceTransferFunction);
        cudaFree(devicePaddedImage);
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        
        return output;
    }

    template <typename PixelT>
    std::string FFTFilter<PixelT>::name() const
    {
        return "FFT Gaussian Lowpass Filter";
    }
    
    template <typename PixelT>
    Architecture FFTFilter<PixelT>::arch() const 
    {
        return Architecture::GPU;
    }
    
    template class FFTFilter<uint8_t>;
    template class FFTFilter<float>;
}
