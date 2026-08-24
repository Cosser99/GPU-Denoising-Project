#include "filters/BilateralFilter.h"
#include "filters/FFTFilter.h"
#include "filters/GaussianFilter.h"
#include "filters/MeanFilter.h"
#include "filters/MedianFilter.h"
#include "utils/ImageIO.h"

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifndef IDL_FILTER_OUTPUT_BACKEND
#define IDL_FILTER_OUTPUT_BACKEND "unknown"
#endif

namespace
{
    using Image = idl::Image<uint8_t>;
    using Filter = idl::Filter<uint8_t>;

    struct FilterScenario
    {
        std::string group;
        std::string name;
        std::unique_ptr<Filter> filter;
    };

    bool fileExists(const std::string& path)
    {
        std::ifstream input(path);
        return input.good();
    }

    std::string parentPath(const std::string& path)
    {
        const size_t separator = path.find_last_of('/');
        if (separator == std::string::npos)
        {
            return "";
        }
        return path.substr(0, separator);
    }

    bool createDirectories(const std::string& path)
    {
        if (path.empty())
        {
            return true;
        }

        std::string current;
        for (size_t index = 0; index < path.size(); index++)
        {
            current += path[index];
            if (path[index] != '/' && index + 1 != path.size())
            {
                continue;
            }
            if (current.empty() || current == "/")
            {
                continue;
            }
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
            {
                return false;
            }
        }
        return true;
    }

    std::string inputPath(const std::string& imagePath)
    {
        if (fileExists(imagePath))
        {
            return imagePath;
        }

        const std::string noisyPath = "data/noisy/" + imagePath;
        if (fileExists(noisyPath))
        {
            return noisyPath;
        }

        return imagePath;
    }

    std::string outputPath(const std::string& imagePath)
    {
        const std::string noisyPrefix = "data/noisy/";
        if (imagePath.rfind(noisyPrefix, 0) == 0)
        {
            return "data/output/" + std::string(IDL_FILTER_OUTPUT_BACKEND) + "/" + imagePath.substr(noisyPrefix.size());
        }
        return "data/output/" + std::string(IDL_FILTER_OUTPUT_BACKEND) + "/" + imagePath;
    }

    std::string outputPath(const std::string& imagePath, const std::string& suffix)
    {
        const std::string destinationPath = outputPath(imagePath);
        const size_t separator = destinationPath.find_last_of('/');
        const size_t filenameStart = separator == std::string::npos ? 0 : separator + 1;
        const size_t dot = destinationPath.find_last_of('.');
        if (dot == std::string::npos || dot < filenameStart)
        {
            return destinationPath + suffix;
        }
        return destinationPath.substr(0, dot) + suffix + destinationPath.substr(dot);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <noisy-image> [all|mean|gaussian|median|bilateral|fft]\n";
        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string sourcePath = inputPath(imagePath);
    const std::string requestedGroup = argc == 3 ? argv[2] : "all";
    if (requestedGroup != "all" && requestedGroup != "mean" && requestedGroup != "gaussian" &&
        requestedGroup != "median" && requestedGroup != "bilateral" && requestedGroup != "fft")
    {
        std::cerr << "Unknown filter group: " << requestedGroup << '\n';
        return 1;
    }

    try
    {
        const Image noisy = idl::ImageIO::load(sourcePath);

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

            const std::string destinationPath = outputPath(imagePath, "_" + filterScenario.name);
            if (!createDirectories(parentPath(destinationPath)))
            {
                std::cerr << "Cannot create output directory: " << parentPath(destinationPath) << '\n';
                return 1;
            }

            const Image filtered = filterScenario.filter->apply(noisy);
            idl::ImageIO::save(filtered, destinationPath);
            std::cout << destinationPath << '\n';
        }
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }

    return 0;
}
