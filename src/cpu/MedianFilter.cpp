#include "filters/MedianFilter.h"
#include "core/Timer.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

// Median filter implemented by using the algorithm of the following paper:
// A Fast Two-Dimensional Median Filtering Algorithm, Thomas S. Huang, George J. Yang, Gregory Y. Tang

namespace idl
{
    template <typename PixelT>
    MedianFilter<PixelT>::MedianFilter(uint32_t m, uint32_t n) :
        _m(m),
        _n(n)
    {}

    template <typename PixelT>
    Image<PixelT> MedianFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        
        Timer timer;
        timer.start();

        const int32_t radiusX = static_cast<int32_t>(_m);
        const int32_t radiusY = static_cast<int32_t>(_n);
        const int32_t width = static_cast<int32_t>(input.width());
        const int32_t height = static_cast<int32_t>(input.height());
        const uint8_t nChannels = input.nChannels();
        if (width == 0 || height == 0 || nChannels == 0)
        {
            return output;
        }

        const auto clampX = [width](int32_t x) { return x < 0 ? 0 : (x >= width ? width - 1 : x); };
        const auto clampY = [height](int32_t y) { return y < 0 ? 0 : (y >= height ? height - 1 : y); };

        if constexpr (std::is_same_v<PixelT, uint8_t>)
        {
            constexpr size_t maxN = 256;
            // making one histogram per channel, initialize to 0
            std::vector<std::array<int, maxN>> histogram(nChannels);
            for (auto& channelHistogram : histogram)
            {
                channelHistogram.fill(0);
            }

            // threshold is the median value
            const uint32_t threshold = ((2 * radiusX + 1) * (2 * radiusY + 1) / 2) + 1;

            // median is the value in the middle of the series
            const auto writeMedian = [&](uint32_t x, uint32_t y)
            {
                for (uint8_t channel = 0; channel < nChannels; ++channel)
                {
                    uint32_t count = 0;
                    size_t i = 0;

                    // series may be something like 00000111225555688899999....
                    // sum the occurrencies of a number and check if count >= threshold
                    // at 0 -> count = 5, at 1 -> count = 5 + 3 etc.
                    while (i < maxN)
                    {
                        count += histogram[channel][i];
                        if (count >= threshold)
                        {
                            break;
                        }
                        ++i;
                    }
                    output.pixel(x, y, channel) = static_cast<PixelT>(i);
                }
            };

            const auto shiftX = [&](int32_t x, int32_t y, int32_t direction)
            {
                const int32_t removeX = clampX(direction == 1 ? x - radiusX - 1 : x + radiusX + 1);
                const int32_t addX = clampX(direction == 1 ? x + radiusX : x - radiusX);
                for (int32_t ky = -radiusY; ky <= radiusY; ++ky)
                {
                    const int32_t yy = clampY(y + ky);
                    for (uint8_t channel = 0; channel < nChannels; ++channel)
                    {
                        --histogram[channel][input.pixel(removeX, yy, channel)];
                        ++histogram[channel][input.pixel(addX, yy, channel)];
                    }
                }
            };

            const auto shiftY = [&](int32_t x, int32_t y)
            {
                const int32_t previousY = clampY(y - radiusY - 1);
                const int32_t nextY = clampY(y + radiusY);
                for (int32_t kx = -radiusX; kx <= radiusX; ++kx)
                {
                    const int32_t xx = clampX(x + kx);
                    for (uint8_t channel = 0; channel < nChannels; ++channel)
                    {
                        --histogram[channel][input.pixel(xx, previousY, channel)];
                        ++histogram[channel][input.pixel(xx, nextY, channel)];
                    }
                }
            };

            // initial window (0, 0)
            for (int32_t ky = -radiusY; ky <= radiusY; ++ky)
            {
                for (int32_t kx = -radiusX; kx <= radiusX; ++kx)
                {
                    for (uint8_t channel = 0; channel < nChannels; ++channel)
                    {
                        ++histogram[channel][input.pixel(clampX(kx), clampY(ky), channel)];
                    }
                }
            }

            int32_t x = 0;
            int32_t y = 0;
            int32_t direction = 1;
            writeMedian(x, y);
            while (true)
            {
                // "snake"-like path -> y = 0, x from 0 to width - 1. y = 1, x from width - 1 to 0, etc.
                // hist needs to be coherent for the whole process
                // if returning to x = 0 every time, hist has to be recreated at each iteration of y
                while ((direction == 1 && x + 1 < width) || (direction == -1 && x > 0))
                {
                    x += direction;
                    shiftX(x, y, direction);
                    writeMedian(x, y);
                }
                if (y + 1 >= height)
                {
                    break;
                }
                ++y;
                shiftY(x, y);
                writeMedian(x, y);
                direction *= -1;
            }
        }
        else
        {
            // Standard Median Algorithm
            const size_t windowSize = static_cast<size_t>(2 * radiusX + 1) * (2 * radiusY + 1);
            std::vector<PixelT> window;
            window.reserve(windowSize);
            for (int32_t y = 0; y < height; ++y)
            {
                for (int32_t x = 0; x < width; ++x)
                {
                    for (uint8_t channel = 0; channel < nChannels; ++channel)
                    {
                        window.clear();
                        for (int32_t ky = -radiusY; ky <= radiusY; ++ky)
                        {
                            for (int32_t kx = -radiusX; kx <= radiusX; ++kx)
                            {
                                window.push_back(input.pixel(clampX(x + kx), clampY(y + ky), channel));
                            }
                        }
                        const auto middle = window.begin() + window.size() / 2;
                        std::nth_element(window.begin(), middle, window.end());
                        output.pixel(x, y, channel) = *middle;
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
    std::string MedianFilter<PixelT>::name() const
    {
        return "Median Filter";
    }

    template <typename PixelT>
    Architecture MedianFilter<PixelT>::arch() const
    {
        return Architecture::CPU;
    }

    template class MedianFilter<uint8_t>;
    template class MedianFilter<float>;
}
