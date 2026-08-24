#pragma once

#include <cstdint>

#include "noise/Noise.h"

namespace idl
{
    template <typename PixelT>
    class PoissonNoise : public Noise<PixelT>
    {
        public:
            explicit PoissonNoise(double intensityScale = 1.0, uint64_t seed = 0);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

        private:
            double _intensityScale; // used for modulate intensity -> higher values means less noise intensity
            uint64_t _seed;
    };
}
