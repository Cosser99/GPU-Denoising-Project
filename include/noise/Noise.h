#pragma once

#include <string>
#include <type_traits>

#include "core/Image.h"

namespace idl
{
    template <typename PixelT>
    class Noise
    {
        public:
            static_assert(std::is_arithmetic_v<PixelT>, "PixelT must be an arithmetic type");

            virtual ~Noise() = default;

            virtual Image<PixelT> apply(const Image<PixelT>& input) const = 0;
            virtual std::string name() const = 0;
    };
}
