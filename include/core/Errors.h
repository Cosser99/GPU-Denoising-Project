#pragma once

#include <string>
#include <stdexcept>

namespace idl
{
    class ImageException : public std::runtime_error
    {
        public:
            explicit ImageException(const std::string& message);
    };

    class ImageNoSizeComparison : public ImageException
    {
        public:
            explicit ImageNoSizeComparison(const std::string& message);
    };

    class ImageTooLittle : public ImageException
    {
        public:
            explicit ImageTooLittle(const std::string& message);
    };

    class ImageDifferentSizes : public ImageException
    {
        public:
            explicit ImageDifferentSizes(const std::string& message);
    };

    class ImageAllocationError : public ImageException
    {
        public:
            explicit ImageAllocationError(const std::string& message);
    };

    class ImageResizeError : public ImageException
    {
        public:
            explicit ImageResizeError(const std::string& message);
    };

    class ImageUnsupportedChannelNumber : public ImageException
    {
        public:
            explicit ImageUnsupportedChannelNumber(const std::string& message);
    };
} // namespace