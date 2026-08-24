#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>
#include <vector>

#include "Errors.h"
#include "Types.h"

namespace idl
{
    template <typename PixelT>
    class Image
    {
        public:
            static_assert(std::is_arithmetic_v<PixelT>, "PixelT must be an arithmetic type");

            Image();

            Image(uint32_t width, uint32_t height, uint8_t nChannels);

            PixelT* data();
            const PixelT* data() const;

            PixelT& pixel(uint32_t x, uint32_t y, uint8_t channel);

            const PixelT& pixel(uint32_t x, uint32_t y, uint8_t channel) const;

            uint32_t width() const;
            uint32_t height() const;
            uint8_t nChannels() const;
            size_t vectorSize() const;

            void resize(uint32_t width, uint32_t height, uint8_t nChannels);

        private:
            void allocate(uint32_t width, uint32_t height, uint8_t nChannels, bool construction)
            {
                try
                {
                    _pixels.resize(static_cast<size_t>(width) * height * nChannels);
                }
                catch (const std::exception& error)
                {
                    if (construction)
                    {
                        throw ImageAllocationError(error.what());
                    }
                    throw ImageResizeError(error.what());
                }
            }

            uint32_t _width;
            uint32_t _height;
            uint8_t _nChannels;
            std::vector<PixelT> _pixels;
    };

    using ByteImage = Image<uint8_t>;
    using FloatImage = Image<float>;
}

#include "Image.tpp"
