#pragma once

#include <cstdint>

#include "core/Filter.h"

namespace idl
{
    // Gaussian lowpass filter implemented in the frequency domain with a 2-D FFT.
    template <typename PixelT>
    class FFTFilter : public Filter<PixelT>
    {
        public:
            // cutoff is D0, in frequency-plane pixels of the padded image.
            explicit FFTFilter(double sigma);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

            Architecture arch() const override;

        private:
            double _sigma;
    };
}
