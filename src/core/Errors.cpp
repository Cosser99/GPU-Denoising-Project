#include "core/Errors.h"

namespace idl
{
    ImageException::ImageException(const std::string& message) :
        std::runtime_error(message)
    {}

    ImageNoSizeComparison::ImageNoSizeComparison(const std::string& message) :
        ImageException("Tried comparison with at least one image with size = 0 (" + message + ")")
    {}

    ImageTooLittle::ImageTooLittle(const std::string& message) :
        ImageException("Image dimension is too small (" + message + ")")
    {}

    ImageDifferentSizes::ImageDifferentSizes(const std::string& message) :
        ImageException("Tried comparison between images with different sizes (" + message + ")")
    {}

    ImageAllocationError::ImageAllocationError(const std::string& message) :
        ImageException("Error in allocating space for the new image (" + message + ")")
    {}

    ImageResizeError::ImageResizeError(const std::string& message) :
        ImageException("Error in resizing the image (" + message + ")")
    {}

    ImageUnsupportedChannelNumber::ImageUnsupportedChannelNumber(const std::string& message) :
        ImageException("Image channel length is unsupported (" + message + ")")
    {}
} // namespace