#include "filters/MedianFilter.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace
{
    constexpr unsigned int kBlockSize = 16;

    __device__ int clampCoordinate(int value, int upperBound)
    {
        return value < 0 ? 0 : (value >= upperBound ? upperBound - 1 : value);
    }

    template <typename PixelT>
    __device__ PixelT computeMedian(const PixelT* input, int x, int y, int width, int height,
                                       int nChannels, int channel, int radiusX, int radiusY)
    {
        const int maxWindowSize = 121; // for max 11 x 11 - cuda needs to know the size before execution 
        int windowSize = 0; // window size can be computed as below, but we use it as index to populate the array
        // const int windowSize = (2 * radiusX + 1) * (2 * radiusY + 1);
        
        PixelT window[maxWindowSize];

        // standard median algorithm
        for (int ky = -radiusY; ky <= radiusY; ky++)
        {
            const int yy = clampCoordinate(y + ky, height);
            for (int kx = -radiusX; kx <= radiusX; kx++)
            {
                const int xx = clampCoordinate(x + kx, width);
                
                window[windowSize++] = input[(static_cast<size_t>(yy) * width + xx) * nChannels + channel];
            }
        }

        const int middle = windowSize / 2;

        // since we need the median, we can sort half vector
        for (int i = 0; i <= middle; i++)
        {
            for (int j = i + 1; j < windowSize; j++)
            {
                if (window[j] < window[i])
                {
                    PixelT tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }

        return window[middle];
    }

    template <typename PixelT>
    __global__ void medianFilterKernel(const PixelT* input, PixelT* output, int width, int height,
                                       int nChannels, int radiusX, int radiusY)
    {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int idy = threadIdx.y + blockIdx.y * blockDim.y;
        if (idx >= width || idy >= height)
        {
            return;
        }

        for (int channel = 0; channel < nChannels; channel++)
        {
            output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] =
                computeMedian(input, idx, idy, width, height, nChannels, channel, radiusX, radiusY);
        }
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

        // allocate device memory
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
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

        // launch kernel
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);
        
        cudaEventRecord(kernelStart);
        medianFilterKernel<<<grid, block>>>(deviceInput, deviceOutput, input.width(), input.height(), input.nChannels(), _m, _n);
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
        
        // "freeing" events
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
    std::string MedianFilter<PixelT>::name() const
    {
        return "Median Filter";
    }
    
    template <typename PixelT>
    Architecture MedianFilter<PixelT>::arch() const
    {
        return Architecture::GPU;
    }
    
    template class MedianFilter<uint8_t>;
    template class MedianFilter<float>;
}
