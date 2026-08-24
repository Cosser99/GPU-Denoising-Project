#include "filters/BilateralFilter.h"
#include "core/Timer.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace idl
{
    template <typename PixelT>
    BilateralFilter<PixelT>::BilateralFilter(double sigmaDomain, double sigmaRange) :
        _sigmaDomain(sigmaDomain),
        _sigmaRange(sigmaRange)
    {
        if (_sigmaDomain <= 0.0 || _sigmaRange <= 0.0)
        {
            throw std::invalid_argument("Bilateral filter sigmas must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> BilateralFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        
        Timer timer;
        timer.start();
        
        const int32_t width = static_cast<int32_t>(input.width());
        const int32_t height = static_cast<int32_t>(input.height());
        const uint8_t nChannels = input.nChannels();

        if (width == 0 || height == 0 || nChannels == 0)
        {
            return output;
        }

        // each output pixel is now function of range and domain 
        const int32_t radius = static_cast<int32_t>(std::ceil(3.0 * _sigmaDomain));
        const uint32_t kernelSize = 2 * radius + 1;
        const double domainFactor = -1.0 / (2.0 * _sigmaDomain * _sigmaDomain);
        const double rangeFactor = -1.0 / (2.0 * _sigmaRange * _sigmaRange);

        // domain weights are computed only once, since their value don't change 
        std::vector<double> domainWeights(static_cast<size_t>(kernelSize) * kernelSize);
        for (int32_t ky = -radius; ky <= radius; ++ky)
        {
            for (int32_t kx = -radius; kx <= radius; ++kx)
            {
                domainWeights[(ky + radius) * kernelSize + kx + radius] =
                    std::exp(static_cast<double>(kx * kx + ky * ky) * domainFactor);
            }
        }

        // replicate padding technique -> if out of range, use the closest valid index
        const auto clampX = [width](int32_t x) { return x < 0 ? 0 : (x >= width ? width - 1 : x); };
        const auto clampY = [height](int32_t y) { return y < 0 ? 0 : (y >= height ? height - 1 : y); };

        for (int32_t y = 0; y < height; ++y)
        {
            for (int32_t x = 0; x < width; ++x)
            {
                std::vector<double> sums(nChannels, 0.0);
                double weightSum = 0.0;

                for (int32_t ky = -radius; ky <= radius; ++ky)
                {
                    const int32_t yy = clampY(y + ky);

                    for (int32_t kx = -radius; kx <= radius; ++kx)
                    {
                        const int32_t xx = clampX(x + kx);
                        double colorDistanceSquared = 0.0;

                        // range distance is computed in the combined space of colors.
                        // computing a separate distance for each color lead to a wrong result (especially on edges)
                        for (uint8_t channel = 0; channel < nChannels; ++channel)
                        {
                            const double difference = static_cast<double>(input.pixel(xx, yy, channel)) -
                                                      static_cast<double>(input.pixel(x, y, channel));
                            colorDistanceSquared += difference * difference;
                        }

                        // total weight is the multiplication of domain weight and range weight
                        // range weight is distance^2 * rangeFactor (as domain weight but with colors distance instead of geometric distance)
                        const double weight = domainWeights[(ky + radius) * kernelSize + kx + radius] *
                                              std::exp(colorDistanceSquared * rangeFactor);
                        weightSum += weight;
                        for (uint8_t channel = 0; channel < nChannels; ++channel)
                        {
                            sums[channel] += weight * static_cast<double>(input.pixel(xx, yy, channel));
                        }
                    }
                }
                for (uint8_t channel = 0; channel < nChannels; ++channel)
                {
                    // result is normalized
                    const double value = sums[channel] / weightSum;
                    if constexpr (std::is_integral_v<PixelT>)
                    {
                        output.pixel(x, y, channel) = static_cast<PixelT>(std::round(value));
                    }
                    else
                    {
                        output.pixel(x, y, channel) = static_cast<PixelT>(value);
                    }
                }
            }
        }

        timer.stop();
        this->setFilterTiming({
            0.0,
            0.0,
            0.0,
            timer.elapsedMs()
        });

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
        return Architecture::CPU;
    }

    template class BilateralFilter<uint8_t>;
    template class BilateralFilter<float>;
}
