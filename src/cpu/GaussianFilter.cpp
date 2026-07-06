#include "filters/GaussianFilter.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace idl
{
    template <typename PixelT>
    GaussianFilter<PixelT>::GaussianFilter(double sigma) : _sigma(sigma)
    {
        if (_sigma <= 0.0)
        {
            throw std::invalid_argument("Gaussian sigma must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> GaussianFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());

        const int32_t radius = static_cast<int32_t>(std::ceil(3.0 * _sigma));
        const uint32_t kernelSize = 2 * radius + 1;
        const int32_t width = static_cast<int32_t>(input.width());
        const int32_t height = static_cast<int32_t>(input.height());

        // kernel size is ceil(6 * sigma) x ceil(6 * sigma)
        // but we add +1 to each dimension to have an odd size
        std::vector<double> kernel(static_cast<size_t>(kernelSize) * kernelSize);
        double kernelSum = 0.0;

        // each weigth is calculated as K * exp(- (s^2 + t^2) / (2 * sigma^2))
        // but we can force K = 1, because its contribute will disappear with the normalization after
        for (int32_t t = -radius; t <= radius; t++)
        {
            for (int32_t s = -radius; s <= radius; s++)
            {
                const double weight = std::exp(-static_cast<double>(s * s + t * t) / (2.0 * _sigma * _sigma));
                kernel[(t + radius) * kernelSize + s + radius] = weight;
                kernelSum += weight;
            }
        }

        // normalization
        for (double& weight : kernel)
        {
            weight /= kernelSum;
        }

        // replicate padding technique -> if out of range, use the closest valid index
        const auto clampX = [width](int32_t x) { return x < 0 ? 0 : (x >= width ? width - 1 : x); };
        const auto clampY = [height](int32_t y) { return y < 0 ? 0 : (y >= height ? height - 1 : y); };

        for (int32_t y = 0; y < height; ++y)
        {
            for (int32_t x = 0; x < width; ++x)
            {
                for (uint8_t channel = 0; channel < input.nChannels(); ++channel)
                {
                    double sum = 0.0;

                    // each pixel in the current window is multiplied for the corresponding weigth of the kernel and summed together
                    for (int32_t ky = -radius; ky <= radius; ++ky)
                    {
                        uint32_t yy = clampY(y + ky);
                        for (int32_t kx = -radius; kx <= radius; ++kx)
                        {
                            uint32_t xx = clampX(x + kx);
                            const double weight = kernel[(ky + radius) * kernelSize + kx + radius];
                            sum += weight * static_cast<double>(input.pixel(xx, yy, channel));
                        }
                    }
                    if constexpr (std::is_integral_v<PixelT>)
                    {
                        output.pixel(x, y, channel) = static_cast<PixelT>(std::round(sum));
                    }
                    else
                    {
                        output.pixel(x, y, channel) = static_cast<PixelT>(sum);
                    }
                }
            }
        }
        return output;
    }

    template <typename PixelT>
    std::string GaussianFilter<PixelT>::name() const
    {
        return "Gaussian Filter";
    }

    template <typename PixelT>
    Architecture GaussianFilter<PixelT>::arch() const
    {
        return Architecture::CPU;
    }

    template class GaussianFilter<uint8_t>;
    template class GaussianFilter<float>;
}
