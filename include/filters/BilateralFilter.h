#pragma once

#include <cstdint>

#include "core/Filter.h"

namespace idl
{
    template <typename PixelT>
    class BilateralFilter : public Filter<PixelT>
    {
        public:
            // sigmaDomain is expressed in pixels; sigmaRange in channel-value units (0-255 for uint8).
            BilateralFilter(double sigmaDomain, double sigmaRange);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

            Architecture arch() const override;

        private:
            double _sigmaDomain;
            double _sigmaRange;
    };
}
