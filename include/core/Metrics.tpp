#pragma once

#include "Errors.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
    // same behavior as fspecial matlab function
    inline std::vector<double> fspecialGaussian(const int size, const double sigma)
    {
        if (size % 2 == 0)
        {
            throw std::invalid_argument("Invalid kernel size: Gaussian kernel must be odd");
        }

        std::vector<double> g(size * size);

        const int start = -size / 2;
        const int end = size / 2;

        double sum = 0.0;

        for (int x = start; x <= end; x++)
        {
            for (int y = start; y <= end; y++)
            {
                g[(x - start) * size + (y - start)] = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                sum += g[(x - start) * size + (y - start)];
            }
        }

        for (int x = start; x <= end; x++)
        {
            for (int y = start; y <= end; y++)
            {
                g[(x - start) * size + (y - start)] /= sum;
            }
        }

        return g;
    }

    // same behavior as conv2 matlab function
    inline std::vector<double> conv2_valid
    (
        const std::vector<double>& img,
        int width,
        int height,
        const std::vector<double>& kernel,
        int size
    )
    {
        int endW = width - size + 1;
        int endH = height - size + 1;

        std::vector<double> out(endW * endH);

        for (int r = 0; r < endH; r++)
        {
            for (int c = 0; c < endW; c++)
            {
                double sum = 0.0;

                for (int rr = 0; rr < size; rr++)
                {
                    for (int cc = 0; cc < size; cc++)
                    {
                        sum +=
                            img[(r + rr) * width + (c + cc)] *
                            kernel[rr * size + cc];
                    }
                }

                out[r * endW + c] = sum;
            }
        }

        return out;
    }
}

namespace idl
{
    template <typename PixelT>
    double Metrics<PixelT>::computeMSE(
        const Image<PixelT>& original,
        const Image<PixelT>& filtered
    )
    {
        // MSE can be computed only on images on the same size
        // (it should be the same just filtered)
        if
        (
            original.vectorSize() != filtered.vectorSize()
            || original.width() != filtered.width()
            || original.height() != filtered.height()
            || original.nChannels() != filtered.nChannels()
        )
        {
            throw ImageDifferentSizes(
                "Original size: " + std::to_string(original.vectorSize()) + 
                " - Filtered size: " + std::to_string(filtered.vectorSize()));
        }

        double error = 0.0;
        double diff;

        const PixelT* originalPixels = original.data();
        const PixelT* filteredPixels = filtered.data();

        const size_t vectorSize = original.vectorSize();

        // MSE = sum((x_i - ^x_i)^2) / n
        for (size_t i = 0; i < vectorSize; i++)
        {
            // to avoid wrap-around, cast values to double
            // using static_cast to do compile-time cast
            diff = static_cast<double>(originalPixels[i]) - static_cast<double>(filteredPixels[i]);
            error += diff * diff;
        }

        return error / static_cast<double>(vectorSize);
    }

    template <typename PixelT>
    double Metrics<PixelT>::computePSNR(
        const Image<PixelT>& original,
        const Image<PixelT>& filtered
    )
    {
        double eps = 1.0e-15;
        double mse = computeMSE(original, filtered);

        // can't compare double directly because of precision loss
        // an epsilon ~ 0 must be used instead of == operator to avoid problems
        if (mse < eps)
        {
            return 100.0;
        }

        // PSNR = 10 * log10 (MAX_VALUE^2 / MSE)
        return 10 * std::log10(maximumValue() * maximumValue() / mse);
    }

    // MSSIM is computed as in https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    template <typename PixelT>
    double Metrics<PixelT>::computeMSSIM(
        const Image<PixelT>& original,
        const Image<PixelT>& filtered
    )
    {
        const int windowSize = 11;
        const int width = original.width();
        const int height = original.height();
        const uint8_t nChannels = original.nChannels();

        // MSSIM can be computed only on images on the same size
        // (it should be the same just filtered)
        if
        (
            original.vectorSize() != filtered.vectorSize()
            || original.width() != filtered.width()
            || original.height() != filtered.height()
            || original.nChannels() != filtered.nChannels()
        )
        {
            throw ImageDifferentSizes("Original size: " + std::to_string(original.vectorSize()) + " - Filtered size: " + std::to_string(filtered.vectorSize()));
        }
        // to apply convolution, image must be at least as big as the filter
        if (width < windowSize || height < windowSize)
        {
            throw ImageTooLittle("Min Dim: " + std::to_string(windowSize) + "x" + std::to_string(windowSize));
        }
        const size_t len = original.vectorSize();

        const PixelT* originalPixels = original.data();
        const PixelT* filteredPixels = filtered.data();
        // converting values into double
        std::vector<double> originalPixelsDouble(len);
        std::vector<double> filteredPixelsDouble(len);

        for (size_t i = 0; i < len; i++)
        {
            originalPixelsDouble[i] = static_cast<double>(originalPixels[i]);
            filteredPixelsDouble[i] = static_cast<double>(filteredPixels[i]);
        }

        const double sigma = 1.5;
        const std::vector<double> window = fspecialGaussian(windowSize, sigma);

        const double k1 = 0.01;
        const double k2 = 0.03;

        const double L = maximumValue();

        const double c1 = (k1 * L) * (k1 * L);
        const double c2 = (k2 * L) * (k2 * L);

        std::vector<double> originalLuma(len / nChannels);
        std::vector<double> filteredLuma(len / nChannels);
        std::vector<double> originalOriginalLuma(width * height);
        std::vector<double> filteredFilteredLuma(width * height);
        std::vector<double> originalFilteredLuma(width * height);

        // before computing the mean and the std, convert RGB to luma using CCIR 601:
        // 0.299 R + 0.587 G + 0.114 B
        if (nChannels == 3 || nChannels == 4)
        {
            // if RGBA, ignore alpha. Only RGB are used to compute luma
            for (size_t i = 0, j = 0; i < len; i += nChannels, j++)
            {
                originalLuma[j] = 0.299 * originalPixelsDouble[i] + 0.587 * originalPixelsDouble[i + 1] + 0.114 * originalPixelsDouble[i + 2];
                filteredLuma[j] = 0.299 * filteredPixelsDouble[i] + 0.587 * filteredPixelsDouble[i + 1] + 0.114 * filteredPixelsDouble[i + 2];
            }
        }
        else if (nChannels == 1)
        {
            for (size_t i = 0, j = 0; i < len; i++, j++)
            {
                originalLuma[i] = originalPixelsDouble[i];
                filteredLuma[i] = filteredPixelsDouble[i];
            }
        }
        else
        {
            throw ImageUnsupportedChannelNumber("image channel number: " + std::to_string(nChannels));
        }

        std::transform(originalLuma.begin(), originalLuma.end(), originalLuma.begin(), originalOriginalLuma.begin(), std::multiplies<double>());
        std::transform(filteredLuma.begin(), filteredLuma.end(), filteredLuma.begin(), filteredFilteredLuma.begin(), std::multiplies<double>());
        std::transform(originalLuma.begin(), originalLuma.end(), filteredLuma.begin(), originalFilteredLuma.begin(), std::multiplies<double>());

        std::vector<double> mu1 = conv2_valid(originalLuma, width, height, window, windowSize);
        std::vector<double> mu2 = conv2_valid(filteredLuma, width, height, window, windowSize);

        size_t muLen = mu1.size();

        std::vector<double> mu11(muLen);
        std::vector<double> mu22(muLen);
        std::vector<double> mu12(muLen);

        std::transform(mu1.begin(), mu1.end(), mu1.begin(), mu11.begin(), std::multiplies<double>());
        std::transform(mu2.begin(), mu2.end(), mu2.begin(), mu22.begin(), std::multiplies<double>());
        std::transform(mu1.begin(), mu1.end(), mu2.begin(), mu12.begin(), std::multiplies<double>());

        std::vector<double> sigma11 = conv2_valid(originalOriginalLuma, width, height, window, windowSize);
        std::vector<double> sigma22 = conv2_valid(filteredFilteredLuma, width, height, window, windowSize);
        std::vector<double> sigma12 = conv2_valid(originalFilteredLuma, width, height, window, windowSize);

        std::transform(sigma11.begin(), sigma11.end(), mu11.begin(), sigma11.begin(), std::minus<double>());
        std::transform(sigma22.begin(), sigma22.end(), mu22.begin(), sigma22.begin(), std::minus<double>());
        std::transform(sigma12.begin(), sigma12.end(), mu12.begin(), sigma12.begin(), std::minus<double>());

        std::vector<double> ssim(muLen);

        for (size_t i = 0; i < muLen; i++)
        {
            ssim[i] = (2 * mu12[i] + c1) * (2 * sigma12[i] + c2) / ((mu11[i] + mu22[i] + c1) * (sigma11[i] + sigma22[i] + c2));
        }

        // apply mean to ssim to have a single value for comparisons
        double mssim = std::accumulate(ssim.begin(), ssim.end(), 0.0) / ssim.size();

        return mssim;
    }
} // namespace
