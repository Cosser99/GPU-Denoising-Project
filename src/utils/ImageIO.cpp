#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"

#include "utils/ImageIO.h"

#include <stdexcept>

namespace idl
{
    Image<uint8_t> ImageIO::load(const std::string& path)
    {
        int width, height, channels;

        unsigned char* data =
            stbi_load(
                path.c_str(),
                &width,
                &height,
                &channels,
                0
            );

        if (!data)
        {
            throw std::runtime_error("Failed to load image");
        }

        Image<uint8_t> image(
            width,
            height,
            channels
        );

        std::copy(
            data,
            data + image.vectorSize(),
            image.data()
        );

        stbi_image_free(data);

        return image;
    }

    void ImageIO::save(
        const Image<uint8_t>& img,
        const std::string& path
    )
    {
        stbi_write_png(
            path.c_str(),
            img.width(),
            img.height(),
            img.nChannels(),
            img.data(),
            img.width() * img.nChannels()
        );
    }
}
