#pragma once

#include <cstdint>

#include "core/Filter.h"

namespace idl
{
    template <typename PixelT>
    class GaussianFilter : public Filter<PixelT>
    {
        public:
            explicit GaussianFilter(double sigma);

            Image<PixelT> apply(const Image<PixelT>& input) const override;

            std::string name() const override;

            Architecture arch() const override;
            
        private:
            double _sigma;
    };

}
