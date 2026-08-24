#pragma once

#include <cstdint>

#include "noise/Noise.h"

namespace idl
{
    template <typename PixelT>
    class GaussianNoise : public Noise<PixelT>
    {
        public:
            GaussianNoise(double mean, double standardDeviation, uint64_t seed = 0);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

        private:
            double _mean; // shift
            double _standardDeviation; // intensity of noise
            uint64_t _seed;
    };
}
