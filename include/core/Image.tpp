#pragma once

#include "Errors.h"

namespace idl
{
    template <typename PixelT>
    Image<PixelT>::Image() : _width(0), _height(0), _nChannels(0) {}

    template <typename PixelT>
    Image<PixelT>::Image(uint32_t width, uint32_t height, uint8_t nChannels) :
        _width(width),
        _height(height),
        _nChannels(nChannels)
    {
        allocate(_width, _height, _nChannels, true);
    }

    template <typename PixelT>
    PixelT* Image<PixelT>::data() { return _pixels.data(); }

    template <typename PixelT>
    const PixelT* Image<PixelT>::data() const { return _pixels.data(); }

    template <typename PixelT>
    PixelT& Image<PixelT>::pixel(uint32_t x, uint32_t y, uint8_t channel)
    {
        return _pixels[(static_cast<size_t>(y) * _width + x) * _nChannels + channel];
    }

    template <typename PixelT>
    const PixelT& Image<PixelT>::pixel(uint32_t x, uint32_t y, uint8_t channel) const
    {
        return _pixels[(static_cast<size_t>(y) * _width + x) * _nChannels + channel];
    }

    template <typename PixelT>
    uint32_t Image<PixelT>::width() const { return _width; }
    
    template <typename PixelT>
    uint32_t Image<PixelT>::height() const { return _height; }
    
    template <typename PixelT>
    uint8_t Image<PixelT>::nChannels() const { return _nChannels; }
    
    template <typename PixelT>
    size_t Image<PixelT>::vectorSize() const { return _pixels.size(); }

    template <typename PixelT>
    void Image<PixelT>::resize(uint32_t width, uint32_t height, uint8_t nChannels)
    {
        if (_width == width && _height == height && _nChannels == nChannels)
        {
            return;
        }

        _width = width;
        _height = height;
        _nChannels = nChannels;
        Image<PixelT>::allocate(_width, _height, _nChannels, false);
    }
}