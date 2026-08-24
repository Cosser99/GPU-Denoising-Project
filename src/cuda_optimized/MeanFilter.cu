#include "filters/MeanFilter.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            std::fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
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
    // optimized version with shared-memory
    // a block loads several pixels in the shared memory, which are used by its threads in parallel 
    __global__ void meanFilterSharedKernel(const PixelT* input, PixelT* output, int width, int height,
                                           int nChannels, int radiusX, int radiusY)
    {
        // dynamic shared memory, its size is computed at runtime
        // unsigned char -> PixelT. with direct PixeT it gives errors... 
        extern __shared__ unsigned char sharedBytes[];
        PixelT* tile = reinterpret_cast<PixelT*>(sharedBytes);

        const int blockX = blockIdx.x * blockDim.x;
        const int blockY = blockIdx.y * blockDim.y;

        const size_t tileWidth = blockDim.x + 2 * radiusX;
        const size_t tileHeight = blockDim.y + 2 * radiusY;

        // threadIdx.y * blockDim.x gives us the row, threadsIdx.x gives us the column
        const int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
        // blockDim.x * blockDim.y tells us how large is the matrix we are considering
        const size_t threadCount = blockDim.x * blockDim.y;
        // tileWidth * tileHeight * nChannels tells us how much elements are in the tile 
        const size_t tileElementCount = tileWidth * tileHeight * nChannels;

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

        const int idx = blockX + threadIdx.x;
        const int idy = blockY + threadIdx.y;

        // out of range
        if (idx >= width || idy >= height)
        {
            return;
        }

        for (int channel = 0; channel < nChannels; channel++)
        {
            double sum = 0.0;
            const unsigned int count = (2 * radiusX + 1) * (2 * radiusY + 1);

            for (int ky = -radiusY; ky <= radiusY; ky++)
            {
                for (int kx = -radiusX; kx <= radiusX; kx++)
                {
                    const int xx = threadIdx.x + kx + radiusX;
                    const int yy = threadIdx.y + ky + radiusY;
                    sum += static_cast<double>(tile[(static_cast<size_t>(yy) * tileWidth + xx) * nChannels + channel]);
                }
            }
            
            if constexpr (std::is_integral_v<PixelT>)
            {
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = 
                    static_cast<PixelT>(floor(sum / count + 0.5));
            }
            else
            {
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = 
                    static_cast<PixelT>(sum / count);
            }
        }
    }

    size_t sharedMemoryBytes(uint32_t radiusX, uint32_t radiusY, uint8_t nChannels, size_t pixelSize)
    {
        // if radius = 1 and width = 8, total width is 8 + 1 (on left/top) + 1 (on right/bottom)
        const size_t tileWidth = kBlockSize + 2 * radiusX;
        const size_t tileHeight = kBlockSize + 2 * radiusY;
        if (tileWidth > std::numeric_limits<size_t>::max() / tileHeight ||
            tileWidth * tileHeight > std::numeric_limits<size_t>::max() / nChannels ||
            tileWidth * tileHeight * nChannels > std::numeric_limits<size_t>::max() / pixelSize)
        {
            return std::numeric_limits<size_t>::max();
        }
        // for computing each pixel we need total width * total height (pixel matrix) * nChannels * pixelSize (uint8_t = 1, float = 4, etc.)
        return tileWidth * tileHeight * nChannels * pixelSize;
    }
}

namespace idl
{
    template <typename PixelT>
    MeanFilter<PixelT>::MeanFilter(uint32_t m, uint32_t n) : _m(m), _n(n) {}

    template <typename PixelT>
    Image<PixelT> MeanFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        PixelT* deviceInput;
        PixelT* deviceOutput;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;

        // allocate device memory
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
        cudaCheckErrors("cudaMalloc failure");
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

        // launch kernel
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((input.width() + block.x - 1) / block.x,
                        (input.height() + block.y - 1) / block.y);
        const size_t tileBytes = sharedMemoryBytes(_m, _n, input.nChannels(), sizeof(PixelT));
        
        cudaEventRecord(kernelStart);
        meanFilterSharedKernel<<<grid, block, tileBytes>>>(deviceInput, deviceOutput, input.width(), input.height(),
                                                           input.nChannels(), _m, _n);
        cudaEventRecord(kernelEnd);
        cudaCheckErrors("kernel launch failure");

        // copy data from device to host
        cudaEventRecord(d2hStart);
        cudaMemcpy(output.data(), deviceOutput, output.vectorSize() * sizeof(PixelT), cudaMemcpyDeviceToHost);
        cudaEventRecord(d2hEnd);
        cudaEventSynchronize(d2hEnd);
        cudaCheckErrors("cudaMemcpy D2H failure");

        float h2dMs, kernelMs, d2hMs, deviceTotalMs;
        cudaEventElapsedTime(&h2dMs, h2dStart, h2dEnd);
        cudaEventElapsedTime(&kernelMs, kernelStart, kernelEnd);
        cudaEventElapsedTime(&d2hMs, d2hStart, d2hEnd);
        cudaEventElapsedTime(&deviceTotalMs, h2dStart, d2hEnd);
        cudaCheckErrors("cudaEventElapsedTime failure");
        this->setFilterTiming({h2dMs, kernelMs, d2hMs, deviceTotalMs});
        cudaEventDestroy(h2dStart);
        cudaEventDestroy(h2dEnd);
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelEnd);
        cudaEventDestroy(d2hStart);
        cudaEventDestroy(d2hEnd);
        cudaCheckErrors("cudaEventDestroy failure");

        // free memory
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        return output;
    }

    template <typename PixelT>
    std::string MeanFilter<PixelT>::name() const
    {
        return "Mean Filter";
    }

    template <typename PixelT>
    Architecture MeanFilter<PixelT>::arch() const
    {
        return Architecture::GPU;
    }

    template class MeanFilter<uint8_t>;
    template class MeanFilter<float>;
}
