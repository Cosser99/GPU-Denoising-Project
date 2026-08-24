#include "filters/BilateralFilter.h"

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

    __device__ int clampCoordinate (int value, int upperBound)
    {
        return value < 0 ? 0 : (value >= upperBound ? upperBound - 1 : value);
    }

    template <typename PixelT>
    __global__ void bilateralFilterKernel (const PixelT* input, PixelT* output, const double* domainWeights,
                                          int width, int height, int nChannels, int radius, int kernelSize,
                                          double rangeFactor)
    {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx >= width || idy >= height)
        {
            return;
        }

        for (int channel = 0; channel < nChannels; channel++)
        {
            double sum = 0.0;
            double weightSum = 0.0;
            for (int ky = -radius; ky <= radius; ky++)
            {
                const int yy = clampCoordinate(idy + ky, height);
                for (int kx = -radius; kx <= radius; kx++)
                {
                    const int xx = clampCoordinate(idx + kx, width);
                    double colorDistanceSquared = 0.0;

                    // range distance is computed in the combined space of colors.
                    // computing a separate distance for each color lead to a wrong result (especially on edges)
                    for (int colorChannel = 0; colorChannel < nChannels; colorChannel++)
                    {
                        const double difference = static_cast<double>(input[(static_cast<size_t>(yy) * width + xx) * nChannels + colorChannel]) -
                                                  static_cast<double>(input[(static_cast<size_t>(idy) * width + idx) * nChannels + colorChannel]);
                        colorDistanceSquared += difference * difference;
                    }

                    // total weight is the multiplication of domain weight and range weight
                    // range weight is distance^2 * rangeFactor (as domain weight but with colors distance instead of geometric distance)
                    const double weight = domainWeights[(ky + radius) * kernelSize + kx + radius] * exp(colorDistanceSquared * rangeFactor);
                    weightSum += weight;
                    sum += weight * static_cast<double>(input[(static_cast<size_t>(yy) * width + xx) * nChannels + channel]);
                }
            }

            // result is normalized
            if constexpr (std::is_integral_v<PixelT>)
            {
                // +0.5 to avoid 7.8 -> 7
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = static_cast<PixelT>(floor(sum / weightSum + 0.5));
            }
            else
            {
                output[(static_cast<size_t>(idy) * width + idx) * nChannels + channel] = static_cast<PixelT>(sum / weightSum);
            }
        }
    }
}

namespace idl
{
    template <typename PixelT>
    BilateralFilter<PixelT>::BilateralFilter (double sigmaDomain, double sigmaRange) :
        _sigmaDomain(sigmaDomain), _sigmaRange(sigmaRange)
    {
        if (_sigmaDomain <= 0.0 || _sigmaRange <= 0.0)
        {
            throw std::invalid_argument("Bilateral filter sigmas must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> BilateralFilter<PixelT>::apply (const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        // each output pixel is now function of range and domain
        const int radius = static_cast<int>(std::ceil(3.0 * _sigmaDomain));
        const int kernelSize = 2 * radius + 1;
        const double domainFactor = -1.0 / (2.0 * _sigmaDomain * _sigmaDomain);
        const double rangeFactor = -1.0 / (2.0 * _sigmaRange * _sigmaRange);

        // domain weights are computed only once, since their value don't change
        std::vector<double> domainWeights(static_cast<size_t>(kernelSize) * kernelSize);
        for (int ky = -radius; ky <= radius; ky++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                domainWeights[(ky + radius) * kernelSize + kx + radius] =
                    std::exp(static_cast<double>(kx * kx + ky * ky) * domainFactor);
            }
        }

        PixelT* deviceInput;
        PixelT* deviceOutput;
        double* deviceDomainWeights;
        cudaEvent_t h2dStart, h2dEnd, kernelStart, kernelEnd, d2hStart, d2hEnd;
        
        // allocate device memory
        cudaMalloc(&deviceInput, input.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceOutput, output.vectorSize() * sizeof(PixelT));
        cudaMalloc(&deviceDomainWeights, domainWeights.size() * sizeof(double));
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
        cudaMemcpy(deviceDomainWeights, domainWeights.data(), domainWeights.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(h2dEnd);
        cudaCheckErrors("cudaMemcpy H2D failure");

        // launch kernel
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid((input.width() + block.x - 1) / block.x, (input.height() + block.y - 1) / block.y);
        cudaEventRecord(kernelStart);
        bilateralFilterKernel<<<grid, block>>>(deviceInput, deviceOutput, deviceDomainWeights, input.width(), input.height(),
                                                input.nChannels(), radius, kernelSize, rangeFactor);
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
        cudaFree(deviceDomainWeights);
        cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
        
        return output;
    }

    template <typename PixelT>
    std::string BilateralFilter<PixelT>::name() const 
    {
        return "Bilateral Filter";
    }

    template <typename PixelT>
    Architecture BilateralFilter<PixelT>::arch() const 
    {
        return Architecture::GPU;
    }
    
    template class BilateralFilter<uint8_t>;
    template class BilateralFilter<float>;
}
