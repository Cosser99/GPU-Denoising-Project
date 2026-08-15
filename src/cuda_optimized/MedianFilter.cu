#include "filters/MedianFilter.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
    __device__ PixelT medianFromTile(const PixelT* tile, int localCenterX, int localCenterY,
                                     int tileWidth, int nChannels, int channel, int radiusX, int radiusY)
    {
        const int windowSize = (2 * radiusX + 1) * (2 * radiusY + 1);
        const int middle = windowSize / 2;

        // Standard Median Algorithm
        // Find the value at the middle position without allocating a window per thread.
        int candidateIndex = 0;
        for (int candidateY = -radiusY; candidateY <= radiusY; ++candidateY)
        {
            for (int candidateX = -radiusX; candidateX <= radiusX; ++candidateX, ++candidateIndex)
            {
                const PixelT candidate = tile[(static_cast<size_t>(localCenterY + candidateY) * tileWidth + localCenterX + candidateX) * nChannels + channel];
                int before = 0;
                int valueIndex = 0;
                for (int ky = -radiusY; ky <= radiusY; ++ky)
                {
                    for (int kx = -radiusX; kx <= radiusX; ++kx, ++valueIndex)
                    {
                        const PixelT value = tile[(static_cast<size_t>(localCenterY + ky) * tileWidth + localCenterX + kx) * nChannels + channel];
                        if (value < candidate || (value == candidate && valueIndex < candidateIndex))
                        {
                            ++before;
                        }
                    }
                }
                if (before == middle)
                {
                    return candidate;
                }
            }
        }
        return PixelT{};
    }

    template <typename PixelT>
    __global__ void medianFilterSharedKernel(const PixelT* input, PixelT* output, int width, int height,
                                             int nChannels, int radiusX, int radiusY)
    {
        // dynamic shared memory, its size is computed at runtime
        // unsigned char -> PixelT. with direct PixeT it gives errors... 
        extern __shared__ unsigned char sharedBytes[];
        PixelT* tile = reinterpret_cast<PixelT*>(sharedBytes);

        const int blockX = blockIdx.x * blockDim.x;
        const int blockY = blockIdx.y * blockDim.y;
        
        const int tileWidth = blockDim.x + 2 * radiusX;
        const int tileHeight = blockDim.y + 2 * radiusY;
        
        const int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
        const int threadCount = blockDim.x * blockDim.y;
        const size_t tileElementCount = static_cast<size_t>(tileWidth) * tileHeight * nChannels;

        // loading element in shared memory
        for (size_t element = threadIndex; element < tileElementCount; element += threadCount)
        {
            const int channel = element % nChannels;
            const size_t localPixel = element / nChannels;
            const int localX = localPixel % tileWidth;
            const int localY = localPixel / tileWidth;
            const int x = clampCoordinate(blockX + localX - radiusX, width);
            const int y = clampCoordinate(blockY + localY - radiusY, height);
            tile[element] = input[(static_cast<size_t>(y) * width + x) * nChannels + channel];
        }
        __syncthreads();

        const int x = blockX + threadIdx.x;
        const int y = blockY + threadIdx.y;
        if (x >= width || y >= height)
        {
            return;
        }

        for (int channel = 0; channel < nChannels; ++channel)
        {
            output[(static_cast<size_t>(y) * width + x) * nChannels + channel] =
                medianFromTile(tile, threadIdx.x + radiusX, threadIdx.y + radiusY,
                               tileWidth, nChannels, channel, radiusX, radiusY);
        }
    }

    size_t sharedMemoryBytes(uint32_t radiusX, uint32_t radiusY, uint8_t nChannels, size_t pixelSize)
    {
        return static_cast<size_t>(kBlockSize + 2 * radiusX) * (kBlockSize + 2 * radiusY) * nChannels * pixelSize;
    }
}

namespace idl
{
    template <typename PixelT>
    MedianFilter<PixelT>::MedianFilter(uint32_t m, uint32_t n) : _m(m), _n(n) {}

    template <typename PixelT>
    Image<PixelT> MedianFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        PixelT* deviceInput;
        PixelT* deviceOutput;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
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
        const dim3 grid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);
        const size_t tileBytes = sharedMemoryBytes(_m, _n, input.nChannels(), sizeof(PixelT));
        cudaEventRecord(kernelStart);
        medianFilterSharedKernel<<<grid, block, tileBytes>>>(deviceInput, deviceOutput, input.width(), input.height(),
                                                             input.nChannels(), _m, _n);
        cudaDeviceSynchronize();
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
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        return output;
    }

    template <typename PixelT> std::string MedianFilter<PixelT>::name() const { return "Median Filter"; }
    template <typename PixelT> Architecture MedianFilter<PixelT>::arch() const { return Architecture::GPU; }
    template class MedianFilter<uint8_t>;
    template class MedianFilter<float>;
}
