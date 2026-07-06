#include "noise/GaussianNoise.h"
#include "noise/PoissonNoise.h"
#include "noise/SaltAndPepperNoise.h"
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

namespace
{
    using Image = idl::Image<uint8_t>;
    using Noise = idl::Noise<uint8_t>;

    struct NoiseScenario
    {
        std::string suffix;
        std::unique_ptr<Noise> noise;
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

        const std::string cleanPath = "data/clean/" + imagePath;
        if (fileExists(cleanPath))
        {
            return cleanPath;
        }

        return imagePath;
    }

    std::string outputPath(const std::string& imagePath)
    {
        const std::string cleanPrefix = "data/clean/";
        if (imagePath.rfind(cleanPrefix, 0) == 0)
        {
            return "data/noisy/" + imagePath.substr(cleanPrefix.size());
        }
        return "data/noisy/" + imagePath;
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
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image-path>\n";
        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string sourcePath = inputPath(imagePath);

    try
    {
        const Image image = idl::ImageIO::load(sourcePath);
        std::vector<NoiseScenario> noises;
        noises.push_back({"_poisson", std::make_unique<idl::PoissonNoise<uint8_t>>(0.5, 1403)});
        noises.push_back({"_snp", std::make_unique<idl::SaltAndPepperNoise<uint8_t>>(0.1, 0.1, 1403)});
        noises.push_back({"_gauss", std::make_unique<idl::GaussianNoise<uint8_t>>(0.0, 50.0, 1403)});

        // all 3 noises are applied to the image
        for (const NoiseScenario& noiseScenario : noises)
        {
            // destination path = original path but in noisy folder instead of original
            const std::string destinationPath = outputPath(imagePath, noiseScenario.suffix);
            const Image noisy = noiseScenario.noise->apply(image);
            if (!createDirectories(parentPath(destinationPath)))
            {
                std::cerr << "Cannot create output directory: " << parentPath(destinationPath) << '\n';
                return 1;
            }
            idl::ImageIO::save(noisy, destinationPath);
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
