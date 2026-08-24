#pragma once

#include <string>

namespace idl // image denoising library
{
    enum class Architecture
    {
        CPU,
        GPU
    };

    struct FilterTiming
    {
        double hostToDeviceMs = 0.0;
        double kernelMs = 0.0;
        double deviceToHostMs = 0.0;
        double totalMs = 0.0;
    };

    struct BenchmarkResult
    {
        std::string filterName;
        Architecture architecture;
        double executionTimeMs;
        double throughputMPs;
        double mse;
        double psnr;
        double mssim;
        FilterTiming filterTiming;
    };
} // namespace
