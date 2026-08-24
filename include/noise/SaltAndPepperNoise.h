#pragma once

#include <cstdint>

#include "noise/Noise.h"

namespace idl
{
    template <typename PixelT>
    class SaltAndPepperNoise : public Noise<PixelT>
    {
        public:
            SaltAndPepperNoise(double saltProbability, double pepperProbability, uint64_t seed = 0);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

        private:
            double _saltProbability;
            double _pepperProbability;
            uint64_t _seed;
    };
}
