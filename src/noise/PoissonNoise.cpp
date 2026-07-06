#include "noise/PoissonNoise.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace
{
    template <typename PixelT>
    PixelT poissonValue(double value)
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
    PoissonNoise<PixelT>::PoissonNoise(double intensityScale, uint64_t seed) : _intensityScale(intensityScale), _seed(seed)
    {
        if (_intensityScale <= 0.0)
        {
            throw std::invalid_argument("Poisson scale must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> PoissonNoise<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        std::mt19937_64 generator(_seed);

        // computing poisson(lambda) for each pixel
        // intensityScale is used to modulate intensity
        for (size_t index = 0; index < input.vectorSize(); ++index)
        {
            // max used for negative floats
            const double lambda = std::max(0.0, static_cast<double>(input.data()[index])) * _intensityScale;
            std::poisson_distribution<uint64_t> distribution(lambda);
            output.data()[index] = poissonValue<PixelT>(static_cast<double>(distribution(generator)) / _intensityScale);
        }
        return output;
    }

    template <typename PixelT>
    std::string PoissonNoise<PixelT>::name() const
    {
        return "Poisson Noise";
    }

    template class PoissonNoise<uint8_t>;
    template class PoissonNoise<float>;
}
