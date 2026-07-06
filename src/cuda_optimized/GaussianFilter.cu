#include "filters/GaussianFilter.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            std::fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            std::fprintf(stderr, "*** FAILED - ABORTING\n"); \
            std::exit(1); \
        } \
    } while (0)

namespace
{
    constexpr unsigned int kBlockSize = 16;

    __device__ int clampCoordinate(int value, int upperBound)
    {
        return value < 0 ? 0 : (value >= upperBound ? upperBound - 1 : value);
    }

    template <typename PixelT>
    __global__ void gaussianFilterSharedKernel(const PixelT* input, PixelT* output, const double* kernel,
                                               int width, int height, int nChannels, int radius, int kernelSize)
    {
        // dynamic shared memory, its size is computed at runtime
        // unsigned char -> PixelT. with direct PixeT it gives errors... 
        extern __shared__ unsigned char sharedBytes[];
        PixelT* tile = reinterpret_cast<PixelT*>(sharedBytes);

        const int blockX = blockIdx.x * blockDim.x;
        const int blockY = blockIdx.y * blockDim.y;
        
        const int tileWidth = blockDim.x + 2 * radius;
        const int tileHeight = blockDim.y + 2 * radius;
        
        // threadIdx.y * blockDim.x gives us the row, threadsIdx.x gives us the column
        const int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
        // blockDim.x * blockDim.y tells us how large is the matrix we are considering
        const int threadCount = blockDim.x * blockDim.y;
        // tileWidth * tileHeight * nChannels tells us how much elements are in the tile 
        const size_t tileElementCount = static_cast<size_t>(tileWidth) * tileHeight * nChannels;

        // loading element in shared memory
        for (size_t element = threadIndex; element < tileElementCount; element += threadCount)
        {
            const int channel = element % nChannels;
            const size_t localPixel = element / nChannels;
            
            const int localX = localPixel % tileWidth;
            const int localY = localPixel / tileWidth;
            
            const int x = clampCoordinate(blockX + localX - radius, width);
            const int y = clampCoordinate(blockY + localY - radius, height);
            tile[element] = input[(static_cast<size_t>(y) * width + x) * nChannels + channel];
        }
        __syncthreads();

        const int idx = blockX + threadIdx.x;
        const int idy = blockY + threadIdx.y;

        // out of range
        if (idx >= width || idy >= height)
        {
            return;
        }

        for (int channel = 0; channel < nChannels; ++channel)
        {
            double sum = 0.0;

            // each pixel in the current window is multiplied for the corresponding weigth of the kernel and summed together
            for (int ky = -radius; ky <= radius; ++ky)
            {
                for (int kx = -radius; kx <= radius; ++kx)
                {
                    const double weight = kernel[(ky + radius) * kernelSize + kx + radius];
                    const int localX = threadIdx.x + kx + radius;
                    const int localY = threadIdx.y + ky + radius;
                    sum += weight * static_cast<double>(tile[(static_cast<size_t>(localY) * tileWidth + localX) * nChannels + channel]);
                }
            }
            if constexpr (std::is_integral_v<PixelT>)
            {
                // since casting truncate the floating part without rounding, adding 0.5 to the value will do the work
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = static_cast<PixelT>(floor(sum + 0.5));
            }
            else
            {
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = static_cast<PixelT>(sum);
            }
        }
    }

    size_t sharedMemoryBytes(int radius, uint8_t nChannels, size_t pixelSize)
    {
        return static_cast<size_t>(kBlockSize + 2 * radius) * (kBlockSize + 2 * radius) * nChannels * pixelSize;
    }
}

namespace idl
{
    template <typename PixelT>
    GaussianFilter<PixelT>::GaussianFilter(double sigma) : _sigma(sigma)
    {
        if (_sigma <= 0.0)
        {
            throw std::invalid_argument("Gaussian sigma must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> GaussianFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        const int radius = static_cast<int>(std::ceil(3.0 * _sigma));
        const int kernelSize = 2 * radius + 1;

        // kernel size is ceil(6 * sigma) x ceil(6 * sigma)
        // but we add +1 to each dimension to have an odd size
        std::vector<double> kernel(static_cast<size_t>(kernelSize) * kernelSize);
        double kernelSum = 0.0;

        // each weigth is calculated as K * exp(- (s^2 + t^2) / (2 * sigma^2))
        // but we can force K = 1, because its contribute will disappear with the normalization after
        for (int t = -radius; t <= radius; ++t)
        {
            for (int s = -radius; s <= radius; ++s)
            {
                const double weight = std::exp(-static_cast<double>(s * s + t * t) / (2.0 * _sigma * _sigma));
                kernel[(t + radius) * kernelSize + s + radius] = weight;
                kernelSum += weight;
            }
        }

        // normalization
        for (double& weight : kernel)
        {
            weight /= kernelSum;
        }

        PixelT* deviceInput;
        PixelT* deviceOutput;
        double* deviceKernel;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceKernel, kernel.size() * sizeof(double));
        cudaCheckErrors("cudaMalloc failure");
        cudaEventCreate(&h2dStart); cudaEventCreate(&h2dEnd);
        cudaEventCreate(&kernelStart); cudaEventCreate(&kernelEnd);
        cudaEventCreate(&d2hStart); cudaEventCreate(&d2hEnd);
        cudaCheckErrors("cudaEventCreate failure");
        cudaEventRecord(h2dStart);
        cudaMemcpy(deviceInput, input.data(), input.vectorSize() * sizeof(PixelT), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceKernel, kernel.data(), kernel.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(h2dEnd);
        cudaCheckErrors("cudaMemcpy H2D failure");

        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);
        const size_t tileBytes = sharedMemoryBytes(radius, input.nChannels(), sizeof(PixelT));
        cudaEventRecord(kernelStart);
        gaussianFilterSharedKernel<<<grid, block, tileBytes>>>(deviceInput, deviceOutput, deviceKernel,
                                                               input.width(), input.height(), input.nChannels(), radius, kernelSize);
        cudaEventRecord(kernelEnd);
        cudaCheckErrors("kernel launch failure");

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
        cudaEventDestroy(h2dStart); cudaEventDestroy(h2dEnd); cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd); cudaEventDestroy(d2hStart); cudaEventDestroy(d2hEnd);
        cudaCheckErrors("cudaEventDestroy failure");
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceKernel);
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        return output;
    }

    template <typename PixelT> std::string GaussianFilter<PixelT>::name() const { return "Gaussian Filter"; }
    template <typename PixelT> Architecture GaussianFilter<PixelT>::arch() const { return Architecture::GPU; }
    template class GaussianFilter<uint8_t>;
    template class GaussianFilter<float>;
}
