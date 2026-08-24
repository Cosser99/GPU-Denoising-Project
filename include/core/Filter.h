#pragma once

#include <cstdint>
#include <string>

#include "Image.h"
#include "Types.h"

namespace idl
{
    template <typename PixelT>
    class Filter
    {
        public:
            static_assert(std::is_arithmetic_v<PixelT>, "PixelT must be an arithmetic type");

            virtual ~Filter() = default;

            virtual Image<PixelT> apply(const Image<PixelT>& input) const = 0;
            virtual std::string name() const = 0;
            virtual Architecture arch() const = 0;

            FilterTiming filterTiming() const
            {
                return _filterTiming;
            }

        protected:
            void setFilterTiming(const FilterTiming& timing) const
            {
                _filterTiming = timing;
            }

        private:
            mutable FilterTiming _filterTiming{};
    };

    using ByteFilter = Filter<uint8_t>;
}
