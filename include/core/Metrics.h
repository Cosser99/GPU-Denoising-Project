#pragma once

#include "Image.h"

#include <limits>
#include <type_traits>

namespace idl
{
    template <typename PixelT>
    class Metrics
    {
        public:
            static_assert(std::is_arithmetic_v<PixelT>, "PixelT must be an arithmetic type");

            static double computeMSE(
                const Image<PixelT>& original,
                const Image<PixelT>& filtered
            );

            static double computePSNR(
                const Image<PixelT>& original,
                const Image<PixelT>& filtered
            );

            static double computeMSSIM(
                const Image<PixelT>& original,
                const Image<PixelT>& filtered
            );

        private:
            static constexpr double maximumValue()
            {
                return static_cast<double>(std::numeric_limits<PixelT>::max());
            }
    };
} // namespace

#include "Metrics.tpp"
