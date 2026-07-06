#pragma once

#include <string>

namespace idl // image denoising library
{
    enum class Architecture
    {
        CPU,
        GPU
    };

    struct GpuTiming
    {
        bool available = false;
        double hostToDeviceMs = 0.0;
        double kernelMs = 0.0;
        double deviceToHostMs = 0.0;
        double deviceTotalMs = 0.0;
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
        GpuTiming gpuTiming;
    };
} // namespace
