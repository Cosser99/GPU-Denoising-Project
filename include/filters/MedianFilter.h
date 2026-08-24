#pragma once

#include <cstdint>

#include "core/Filter.h"

namespace idl
{
    template <typename PixelT>
    class MedianFilter : public Filter<PixelT>
    {
        public:
            explicit MedianFilter(uint32_t m, uint32_t n);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

            Architecture arch() const override;
            
        private:
            uint32_t _m;
            uint32_t _n;
    };

}
