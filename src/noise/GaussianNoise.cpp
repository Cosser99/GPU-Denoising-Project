#include "noise/GaussianNoise.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace
{
    template <typename PixelT>
    PixelT gaussianValue(double value)
    {
        if constexpr (std::is_integral_v<PixelT>)
        {
            const double rounded = std::round(value);
            return static_cast<PixelT>(std::clamp(
                rounded,
                static_cast<double>(std::numeric_limits<PixelT>::lowest()),
                static_cast<double>(std::numeric_limits<PixelT>::max())
            ));
        }
        else
        {
            return static_cast<PixelT>(value);
        }
    }
}

namespace idl
{
    template <typename PixelT>
    GaussianNoise<PixelT>::GaussianNoise(double mean, double standardDeviation, uint64_t seed) :
        _mean(mean),
        _standardDeviation(standardDeviation),
        _seed(seed)
    {
        if (_standardDeviation <= 0.0)
        {
            throw std::invalid_argument("Gaussian standard deviation must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> GaussianNoise<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        std::mt19937_64 generator(_seed);
        std::normal_distribution<double> distribution(_mean, _standardDeviation);

        // gaussian noise is additive: g(x, y) = f(x, y) + h(x, y).
        for (size_t index = 0; index < input.vectorSize(); ++index)
        {
            output.data()[index] = gaussianValue<PixelT>(static_cast<double>(input.data()[index]) + distribution(generator));
        }
        return output;
    }

    template <typename PixelT>
    std::string GaussianNoise<PixelT>::name() const
    {
        return "Gaussian Noise";
    }

    template class GaussianNoise<uint8_t>;
    template class GaussianNoise<float>;
}
