#pragma once

#include "Errors.h"
#include "Metrics.h"
#include "Timer.h"

#include <stdexcept>

namespace idl
{
    template <typename PixelT>
    BenchmarkResult Benchmark<PixelT>::run(
        const Filter<PixelT>& filter,
        const Image<PixelT>& input,
        const Image<PixelT>& original,
        uint32_t repetitions
    )
    {
        if (input.width() != original.width() ||
            input.height() != original.height() ||
            input.nChannels() != original.nChannels())
        {
            throw ImageDifferentSizes("Input and original images have different shapes");
        }
        if (repetitions == 0)
        {
            throw std::invalid_argument("Benchmark repetitions must be positive");
        }

        Timer timer;
        BenchmarkResult results{};
        const double imageMegapixels = static_cast<double>(input.width()) * input.height() / 1.0e6;
        Image<PixelT> output;
        GpuTiming totalGpuTiming{};

        timer.start();
        for (uint32_t iteration = 0; iteration < repetitions; ++iteration)
        {
            output = filter.apply(input);
            const GpuTiming gpuTiming = filter.gpuTiming();
            if (gpuTiming.available)
            {
                totalGpuTiming.available = true;
                totalGpuTiming.hostToDeviceMs += gpuTiming.hostToDeviceMs;
                totalGpuTiming.kernelMs += gpuTiming.kernelMs;
                totalGpuTiming.deviceToHostMs += gpuTiming.deviceToHostMs;
                totalGpuTiming.deviceTotalMs += gpuTiming.deviceTotalMs;
            }
        }
        timer.stop();
        results.filterName = filter.name();
        results.architecture = filter.arch();
        results.executionTimeMs = timer.elapsedMs() / repetitions;
        results.throughputMPs = results.executionTimeMs > 0.0
            ? imageMegapixels / results.executionTimeMs * 1000.0
            : 0.0;
        results.mse = Metrics<PixelT>::computeMSE(original, output);
        results.psnr = Metrics<PixelT>::computePSNR(original, output);
        results.mssim = Metrics<PixelT>::computeMSSIM(original, output);
        if (totalGpuTiming.available)
        {
            totalGpuTiming.hostToDeviceMs /= repetitions;
            totalGpuTiming.kernelMs /= repetitions;
            totalGpuTiming.deviceToHostMs /= repetitions;
            totalGpuTiming.deviceTotalMs /= repetitions;
        }
        results.gpuTiming = totalGpuTiming;
        return results;
    }
}
