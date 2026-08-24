#include "filters/MeanFilter.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    __global__ void meanFilterKernel(const PixelT* input, PixelT* output, int width, int height,
                                     int nChannels, int radiusX, int radiusY)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx >= width || idy >= height)
        {
            return;
        }

        const unsigned int count = static_cast<unsigned int>((2 * radiusX + 1) * (2 * radiusY + 1));
        for (int channel = 0; channel < nChannels; channel++)
        {
            double sum = 0.0;
            
            // pixel[i] = 1/(n*m) * sum_{j in range}(k[j])
            for (int ky = -radiusY; ky <= radiusY; ky++)
            {
                // replicate padding technique -> if out of range, use the closest valid index
                const int yy = clampCoordinate(idy + ky, height);
                for (int kx = -radiusX; kx <= radiusX; kx++)
                {
                    const int xx = clampCoordinate(idx + kx, width);
                    sum += static_cast<double>(input[(static_cast<size_t>(yy) * width + xx) * nChannels + channel]);
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
        const dim3 grid((input.width() + block.x - 1) / block.x,
                        (input.height() + block.y - 1) / block.y);
        
        cudaEventRecord(kernelStart);
        meanFilterKernel<<<grid, block>>>(deviceInput, deviceOutput, input.width(), input.height(),
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
