#include "core/Benchmark.h"
#include "filters/BilateralFilter.h"
#include "filters/FFTFilter.h"
#include "filters/GaussianFilter.h"
#include "filters/MeanFilter.h"
#include "filters/MedianFilter.h"
#include "noise/GaussianNoise.h"
#include "noise/PoissonNoise.h"
#include "noise/SaltAndPepperNoise.h"
#include "utils/ImageIO.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef IDL_BENCHMARK_BACKEND
#define IDL_BENCHMARK_BACKEND "unknown"
#endif

namespace
{
    using Image = idl::Image<uint8_t>;
    using Filter = idl::Filter<uint8_t>;
    using Noise = idl::Noise<uint8_t>;

    struct NoiseScenario
    {
        std::string name;
        std::unique_ptr<Noise> noise;
    };

    struct FilterScenario
    {
        std::string group;
        std::string name;
        std::unique_ptr<Filter> filter;
    };

    // writing results in a csv file to easier comparison
    void writeResult(std::ofstream& output, const std::string& noiseName,
                     const std::string& filterName, const idl::BenchmarkResult& result)
    {
        output << IDL_BENCHMARK_BACKEND << ','
               << noiseName << ','
               << filterName << ','
               << std::fixed << std::setprecision(6)
               << result.executionTimeMs << ','
               << result.throughputMPs << ','
               << result.filterTiming.hostToDeviceMs << ','
               << result.filterTiming.kernelMs << ','
               << result.filterTiming.deviceToHostMs << ','
               << result.filterTiming.totalMs << ','
               << result.mse << ','
               << result.psnr << ','
               << result.mssim << '\n';
        output.flush();
    }

    uint32_t parseRepetitions(const char* value)
    {
        const long repetitions = std::strtol(value, nullptr, 10);
        if (repetitions <= 0)
        {
            throw std::invalid_argument("Repetitions must be positive");
        }
        return static_cast<uint32_t>(repetitions);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <clean-image> <results.csv> [repetitions] [all|mean|gaussian|median|bilateral|fft]\n";
        return 1;
    }

    const uint32_t repetitions = argc >= 4 ? parseRepetitions(argv[3]) : 10;
    const std::string requestedGroup = argc == 5 ? argv[4] : "all";
    if (requestedGroup != "all" && requestedGroup != "mean" && requestedGroup != "gaussian" &&
        requestedGroup != "median" && requestedGroup != "bilateral" && requestedGroup != "fft")
    {
        std::cerr << "Unknown filter group: " << requestedGroup << '\n';
        return 1;
    }
    const Image original = idl::ImageIO::load(argv[1]);
    std::ofstream output(argv[2]);
    if (!output)
    {
        std::cerr << "Cannot open CSV output file: " << argv[2] << '\n';
        return 1;
    }

    output << "backend,noise,filter,execution_time_ms,throughput_mps,h2d_ms,kernel_ms,d2h_ms,filter_total_ms,mse,psnr,mssim\n";

    std::vector<NoiseScenario> noises;
    noises.push_back({"poisson_05", std::make_unique<idl::PoissonNoise<uint8_t>>(0.5, 1403)});
    noises.push_back({"saltnpepper_01_01", std::make_unique<idl::SaltAndPepperNoise<uint8_t>>(0.1, 0.1, 1403)});
    noises.push_back({"gaussian_50", std::make_unique<idl::GaussianNoise<uint8_t>>(0.0, 50.0, 1403)});

    // for each noise, apply 2 versions of each filter
    for (const NoiseScenario& noiseScenario : noises)
    {
        const Image noisy = noiseScenario.noise->apply(original);

        std::vector<FilterScenario> filters;
        filters.push_back({"mean", "mean_radius_1x1", std::make_unique<idl::MeanFilter<uint8_t>>(1, 1)});
        filters.push_back({"mean", "mean_radius_5x5", std::make_unique<idl::MeanFilter<uint8_t>>(5, 5)});
        filters.push_back({"gaussian", "gaussian_sigma_1", std::make_unique<idl::GaussianFilter<uint8_t>>(1.0)});
        filters.push_back({"gaussian", "gaussian_sigma_3", std::make_unique<idl::GaussianFilter<uint8_t>>(3.0)});
        filters.push_back({"median", "median_radius_1x1", std::make_unique<idl::MedianFilter<uint8_t>>(1, 1)});
        filters.push_back({"median", "median_radius_5x5", std::make_unique<idl::MedianFilter<uint8_t>>(5, 5)});
        filters.push_back({"bilateral", "bilateral_domain_1_range_20", std::make_unique<idl::BilateralFilter<uint8_t>>(1.0, 20.0)});
        filters.push_back({"bilateral", "bilateral_domain_3_range_50", std::make_unique<idl::BilateralFilter<uint8_t>>(3.0, 50.0)});
        filters.push_back({"fft", "fft_sigma_10", std::make_unique<idl::FFTFilter<uint8_t>>(10.0)});
        filters.push_back({"fft", "fft_sigma_50", std::make_unique<idl::FFTFilter<uint8_t>>(50.0)});

        for (const FilterScenario& filterScenario : filters)
        {
            if (requestedGroup != "all" && requestedGroup != filterScenario.group)
            {
                continue;
            }
            idl::Benchmark<uint8_t> benchmark;
            const idl::BenchmarkResult result = benchmark.run(*filterScenario.filter, noisy, original, repetitions);
            writeResult(output, noiseScenario.name, filterScenario.name, result);
            std::cout << IDL_BENCHMARK_BACKEND << " | " << noiseScenario.name << " | "
                      << filterScenario.name << " | " << result.executionTimeMs << " ms\n";
        }
    }

    return 0;
}
