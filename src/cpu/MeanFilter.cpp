#include "filters/MeanFilter.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace idl
{
    template <typename PixelT>
    MeanFilter<PixelT>::MeanFilter(uint32_t m, uint32_t n) :
        _m(m),
        _n(n)
    {}

    template <typename PixelT>
    Image<PixelT> MeanFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        
        const int32_t radiusX = static_cast<int32_t>(_m);
        const int32_t radiusY = static_cast<int32_t>(_n);
        const int32_t width = static_cast<int32_t>(input.width());
        const int32_t height = static_cast<int32_t>(input.height());

        const auto clampX = [width](int32_t x) { return x < 0 ? 0 : (x >= width ? width - 1 : x); };
        const auto clampY = [height](int32_t y) { return y < 0 ? 0 : (y >= height ? height - 1 : y); };

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                for (uint8_t channel = 0; channel < input.nChannels(); channel++)
                {
                    double sum = 0.0;
                    uint32_t count = 0;

                    // pixel[i] = 1/(n*m) * sum_{j in range}(k[j])
                    for (int32_t ky = -radiusY; ky <= radiusY; ky++)
                    {
                        // replicate padding technique -> if out of range, use the closest valid index
                        const int32_t yy = clampY(y + ky);
                        for (int32_t kx = -radiusX; kx <= radiusX; kx++)
                        {
                            const int32_t xx = clampX(x + kx);
                            sum += static_cast<double>(input.pixel(xx, yy, channel));
                            count++;
                        }
                    }
                    const double value = sum / count;
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
        return Architecture::CPU;
    }

    template class MeanFilter<uint8_t>;
    template class MeanFilter<float>;
}
