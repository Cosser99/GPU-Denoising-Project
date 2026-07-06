#include "noise/SaltAndPepperNoise.h"

#include <limits>
#include <random>
#include <stdexcept>

namespace idl
{
    template <typename PixelT>
    SaltAndPepperNoise<PixelT>::SaltAndPepperNoise(double saltProbability, double pepperProbability, uint64_t seed) :
        _saltProbability(saltProbability),
        _pepperProbability(pepperProbability),
        _seed(seed)
    {
        if (_saltProbability < 0.0 || _pepperProbability < 0.0 ||
            _saltProbability + _pepperProbability > 1.0)
        {
            throw std::invalid_argument("Salt and pepper probabilities must be non-negative and sum to at most one");
        }
    }

    template <typename PixelT>
    Image<PixelT> SaltAndPepperNoise<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        std::mt19937_64 generator(_seed);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        //               [ Ps            z = 2^k - 1
        // PDF -> p(z) = | Pp            z = 0
        //               [ 1 - (Ps + Pp) 0 < z < 2^k -1
        for (size_t index = 0; index < input.vectorSize(); ++index)
        {
            const double probability = distribution(generator);
            // applying pepper noise
            if (probability < _pepperProbability)
            {
                output.data()[index] = static_cast<PixelT>(0);
            }
            // applying salt noise
            else if (probability < _pepperProbability + _saltProbability)
            {
                output.data()[index] = std::numeric_limits<PixelT>::max();
            }
            // no noise with probability 1 - (Ps + Pp)
            else
            {
                output.data()[index] = input.data()[index];
            }
        }
        return output;
    }

    template <typename PixelT>
    std::string SaltAndPepperNoise<PixelT>::name() const
    {
        return "Salt and Pepper Noise";
    }

    template class SaltAndPepperNoise<uint8_t>;
    template class SaltAndPepperNoise<float>;
}
