#pragma once

#include "core/Image.h"
#include <string>

namespace idl
{
    class ImageIO
    {
    public:
        static Image<uint8_t> load(const std::string& path);
        static void save(const Image<uint8_t>& img, const std::string& path);
    };
}
