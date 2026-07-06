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

            GpuTiming gpuTiming() const
            {
                return _gpuTiming;
            }

        protected:
            void setGpuTiming(const GpuTiming& timing) const
            {
                _gpuTiming = timing;
            }

        private:
            mutable GpuTiming _gpuTiming{};
    };

    using ByteFilter = Filter<uint8_t>;
}
