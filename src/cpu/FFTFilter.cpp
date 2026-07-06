#define _USE_MATH_DEFINES

// this algorithm follows the steps of the book cited in the abstract

#include "filters/FFTFilter.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{
    using Complex = std::complex<double>;

    // bit left shift corresponds to multiplication x2
    size_t nextPowerOfTwo(size_t value)
    {
        size_t result = 1;
        while (result < value)
        {
            result <<= 1;
        }
        return result;
    }

    // successive-doubling method
    void fftRecursive(std::vector<Complex>& values)
    {
        const size_t M = values.size();

        // DFT of a single point is the point itself
        if (M <= 1)
        {
            return;
        }

        const size_t K = M / 2;
        std::vector<Complex> even(K);
        std::vector<Complex> odd(K);

        // dividing values in odd samples and even sample
        for (size_t x = 0; x < K; x++)
        {
            even[x] = values[2 * x];
            odd[x] = values[2 * x + 1];
        }

        // recursive calls, from ... 8 to 4, from 4 to 2, from 2 to 1 and come back
        fftRecursive(even);
        fftRecursive(odd);

        // once recursion is finished, we apply
        // F(u) = E(u) + W_M^u * O(u)
        // F(u + K) = E(u) - W_M^u * O(u)
        // with u = 0, ..., K-1
        const double angle = -2.0 * M_PI / static_cast<double>(M);
        const Complex root(std::cos(angle), std::sin(angle));

        // factor = W_M^u, = 1 for the first iteration (W_M^0 = 1)
        Complex factor(1.0, 0.0);
        for (size_t u = 0; u < K; u++)
        {
            const Complex weightedOdd = factor * odd[u];
            values[u] = even[u] + weightedOdd;
            values[u + K] = even[u] - weightedOdd;
            factor *= root;
        }
    }

    void fft1D(std::vector<Complex>& values, bool inverse)
    {
        // if inverse == false -> DFT
        // if inverse == true -> IDFT

        // IDFT(F) = conj(DFT(conf(F))) / Nsamples
        if (inverse)
        {
            for (Complex& value : values)
            {
                value = std::conj(value);
            }
        }

        fftRecursive(values);

        if (inverse)
        {
            for (Complex& value : values)
            {
                value = std::conj(value) / static_cast<double>(values.size());
            }
        }
    }

    void fft2D(std::vector<Complex>& values, size_t width, size_t height, bool inverse)
    {
        std::vector<Complex> row(width);
        std::vector<Complex> column(height);

        // 2d fft can be derived by applying 1d fft on rows and then 1d fft on columns of the result
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                row[x] = values[y * width + x];
            }
            fft1D(row, inverse);
            for (size_t x = 0; x < width; x++)
            {
                values[y * width + x] = row[x];
            }
        }
        for (size_t x = 0; x < width; x++)
        {
            for (size_t y = 0; y < height; y++)
            {
                column[y] = values[y * width + x];
            }
            fft1D(column, inverse);
            for (size_t y = 0; y < height; y++)
            {
                values[y * width + x] = column[y];
            }
        }
    }
}

namespace idl
{
    template <typename PixelT>
    FFTFilter<PixelT>::FFTFilter(double sigma) : _sigma(sigma)
    {
        if (_sigma <= 0.0)
        {
            throw std::invalid_argument("sigma must be positive");
        }
    }

    template <typename PixelT>
    Image<PixelT> FFTFilter<PixelT>::apply(const Image<PixelT>& input) const
    {
        Image<PixelT> output(input.width(), input.height(), input.nChannels());
        if (input.width() == 0 || input.height() == 0 || input.nChannels() == 0)
        {
            return output;
        }

        // Step 1. starting from an input image M x N retrieve padding sizes P = 2M and Q = 2N
        // actually, since successive-doubling method require a power of 2, P and Q are the closest power of 2 >= 2M and 2N
        const size_t paddedWidth = nextPowerOfTwo(2 * static_cast<size_t>(input.width()));
        const size_t paddedHeight = nextPowerOfTwo(2 * static_cast<size_t>(input.height()));

        // symmetric filter transfer function is centered in P/2 Q/2
        const size_t centerX = paddedWidth / 2;
        const size_t centerY = paddedHeight / 2;

        // Step 5. symmetric filter transfer function P x Q
        // made just once (it is the same for each channel, no need to recompute it in each loop)
        std::vector<double> transferFunction(paddedWidth * paddedHeight);
        // gaussian lowpass filter has been chosen
        // other choices are ideal lowpass filter and butterworth lowpass filter, and others

        for (size_t y = 0; y < paddedHeight; ++y)
        {
            const double dy = static_cast<double>(y) - centerY;
            for (size_t x = 0; x < paddedWidth; ++x)
            {
                const double dx = static_cast<double>(x) - centerX;
                // H(u, v) = e^(- D^2(u, v) / (2 * sigma^2) )
                transferFunction[y * paddedWidth + x] = std::exp(-(dx * dx + dy * dy) / (2.0 * _sigma * _sigma));
            }
        }

        // each channel operates independently
        for (uint8_t channel = 0; channel < input.nChannels(); ++channel)
        {
            std::vector<Complex> paddedImage(paddedWidth * paddedHeight);

            // Step 2. make a new image P x Q using replicate padding.
            // original image is located on the top-left corner
            for (size_t y = 0; y < paddedHeight; ++y)
            {
                const uint32_t yy = y < input.height() ? static_cast<uint32_t>(y) : input.height() - 1;
                for (size_t x = 0; x < paddedWidth; ++x)
                {
                    const uint32_t xx = x < input.width() ? static_cast<uint32_t>(x) : input.width() - 1;
                    const double value = static_cast<double>(input.pixel(xx, yy, channel));
                    // Step 3. multiply the image by (-1)^(x+y)
                    paddedImage[y * paddedWidth + x] = ((x + y) % 2 == 0) ? value : -value;
                }
            }

            // Step 4. compute the DFT
            fft2D(paddedImage, paddedWidth, paddedHeight, false);

            for (size_t i = 0; i < paddedImage.size(); i++)
            {
                // Step 6. element wise multiplication H(u,v) x F(u, v) = G(u, v)
                paddedImage[i] *= transferFunction[i];
            }

            // step 7. filtered image is IDFT(G)
            fft2D(paddedImage, paddedWidth, paddedHeight, true);

            // step 8. extract the filtered image from top left corner M x N
            for (size_t y = 0; y < input.height(); ++y)
            {
                for (size_t x = 0; x < input.width(); ++x)
                {
                    double value = paddedImage[y * paddedWidth + x].real();
                    if ((x + y) % 2 != 0)
                    {
                        value = -value;
                    }
                    // if integer, clamp first. if float, just cast
                    if constexpr (std::is_integral_v<PixelT>)
                    {
                        const double rounded = std::round(value);
                        output.pixel(x, y, channel) = static_cast<PixelT>(std::clamp(
                            rounded,
                            static_cast<double>(std::numeric_limits<PixelT>::lowest()),
                            static_cast<double>(std::numeric_limits<PixelT>::max())
                        ));
                    }
                    else
                    {
                        output.pixel(x, y, channel) = static_cast<PixelT>(value);
                    }
                }
            }
        }
        return output;
    }

    template <typename PixelT>
    std::string FFTFilter<PixelT>::name() const
    {
        return "FFT Gaussian Lowpass Filter";
    }

    template <typename PixelT>
    Architecture FFTFilter<PixelT>::arch() const
    {
        return Architecture::CPU;
    }

    template class FFTFilter<uint8_t>;
    template class FFTFilter<float>;
}
