#include <opencv2/opencv.hpp>

#ifdef IDL_OPENCV_USE_CUDA
#if !__has_include(<opencv2/cudaarithm.hpp>) || !__has_include(<opencv2/cudafilters.hpp>) || \
    !__has_include(<opencv2/cudaimgproc.hpp>)
#error "IDL_OPENCV_USE_CUDA requires OpenCV built with CUDA modules: cudaarithm, cudafilters, cudaimgproc."
#else
#define IDL_OPENCV_HAS_CUDA_MODULES 1
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef IDL_OPENCV_BENCHMARK_BACKEND
#ifdef IDL_OPENCV_USE_CUDA
#define IDL_OPENCV_BENCHMARK_BACKEND "opencv_cuda"
#else
#define IDL_OPENCV_BENCHMARK_BACKEND "opencv_cpu"
#endif
#endif

namespace
{
    using Clock = std::chrono::high_resolution_clock;

    struct GpuTiming
    {
        bool available = false;
        double hostToDeviceMs = 0.0;
        double kernelMs = 0.0;
        double deviceToHostMs = 0.0;
        double deviceTotalMs = 0.0;
    };

    struct BenchmarkResult
    {
        double executionTimeMs = 0.0;
        double throughputMPs = 0.0;
        GpuTiming gpuTiming;
        double mse = 0.0;
        double psnr = 0.0;
        double mssim = 0.0;
    };

    struct FilterRun
    {
        cv::Mat output;
        GpuTiming gpuTiming;
    };

    struct NoiseScenario
    {
        std::string name;
        std::function<cv::Mat(const cv::Mat&)> apply;
    };

    struct FilterScenario
    {
        std::string group;
        std::string name;
        std::function<FilterRun(const cv::Mat&)> apply;
    };

    double elapsedMs(const Clock::time_point& start, const Clock::time_point& end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    uint32_t parseRepetitions(const char* value)
    {
        const long repetitions = std::strtol(value, nullptr, 10);
        if (repetitions <= 0)
        {
            throw std::invalid_argument("Repetitions must be positive");
        }
        return static_cast<uint32_t>(repetitions);
    }

    uint8_t toByte(double value)
    {
        const double rounded = std::round(value);
        return static_cast<uint8_t>(std::clamp(rounded, 0.0, 255.0));
    }

    cv::Mat loadImage(const std::string& path)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            throw std::runtime_error("Failed to load image: " + path);
        }
        if (image.depth() != CV_8U)
        {
            throw std::runtime_error("Only 8-bit images are supported");
        }

        cv::Mat converted;
        if (image.channels() == 3)
        {
            cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, converted, cv::COLOR_BGRA2RGBA);
        }
        else if (image.channels() == 1)
        {
            converted = image;
        }
        else
        {
            throw std::runtime_error("Unsupported channel count: " + std::to_string(image.channels()));
        }

        return converted.isContinuous() ? converted.clone() : converted;
    }

    void ensureSameShape(const cv::Mat& original, const cv::Mat& filtered)
    {
        if (original.size() != filtered.size() || original.type() != filtered.type())
        {
            throw std::runtime_error("Input and output images have different shapes");
        }
    }

    cv::Mat applyPoissonNoise(const cv::Mat& input, double intensityScale, uint64_t seed)
    {
        if (intensityScale <= 0.0)
        {
            throw std::invalid_argument("Poisson scale must be positive");
        }

        cv::Mat output(input.size(), input.type());
        std::mt19937_64 generator(seed);

        const size_t count = input.total() * static_cast<size_t>(input.channels());
        const uint8_t* src = input.ptr<uint8_t>(0);
        uint8_t* dst = output.ptr<uint8_t>(0);
        for (size_t index = 0; index < count; ++index)
        {
            const double lambda = static_cast<double>(src[index]) * intensityScale;
            std::poisson_distribution<uint64_t> distribution(lambda);
            dst[index] = toByte(static_cast<double>(distribution(generator)) / intensityScale);
        }
        return output;
    }

    cv::Mat applySaltAndPepperNoise(const cv::Mat& input, double saltProbability,
                                    double pepperProbability, uint64_t seed)
    {
        if (saltProbability < 0.0 || pepperProbability < 0.0 ||
            saltProbability + pepperProbability > 1.0)
        {
            throw std::invalid_argument("Salt and pepper probabilities must be valid");
        }

        cv::Mat output(input.size(), input.type());
        std::mt19937_64 generator(seed);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        const size_t count = input.total() * static_cast<size_t>(input.channels());
        const uint8_t* src = input.ptr<uint8_t>(0);
        uint8_t* dst = output.ptr<uint8_t>(0);
        for (size_t index = 0; index < count; ++index)
        {
            const double probability = distribution(generator);
            if (probability < pepperProbability)
            {
                dst[index] = 0;
            }
            else if (probability < pepperProbability + saltProbability)
            {
                dst[index] = std::numeric_limits<uint8_t>::max();
            }
            else
            {
                dst[index] = src[index];
            }
        }
        return output;
    }

    cv::Mat applyGaussianNoise(const cv::Mat& input, double mean, double standardDeviation, uint64_t seed)
    {
        if (standardDeviation <= 0.0)
        {
            throw std::invalid_argument("Gaussian standard deviation must be positive");
        }

        cv::Mat output(input.size(), input.type());
        std::mt19937_64 generator(seed);
        std::normal_distribution<double> distribution(mean, standardDeviation);

        const size_t count = input.total() * static_cast<size_t>(input.channels());
        const uint8_t* src = input.ptr<uint8_t>(0);
        uint8_t* dst = output.ptr<uint8_t>(0);
        for (size_t index = 0; index < count; ++index)
        {
            dst[index] = toByte(static_cast<double>(src[index]) + distribution(generator));
        }
        return output;
    }

    double computeMSE(const cv::Mat& original, const cv::Mat& filtered)
    {
        ensureSameShape(original, filtered);

        cv::Mat original64;
        cv::Mat filtered64;
        original.convertTo(original64, CV_64F);
        filtered.convertTo(filtered64, CV_64F);

        cv::Mat diff;
        cv::absdiff(original64, filtered64, diff);
        diff = diff.mul(diff);

        const cv::Scalar sum = cv::sum(diff);
        double total = 0.0;
        for (int channel = 0; channel < original.channels(); ++channel)
        {
            total += sum[channel];
        }
        return total / static_cast<double>(original.total() * original.channels());
    }

    double computePSNR(const cv::Mat& original, const cv::Mat& filtered)
    {
        const double mse = computeMSE(original, filtered);
        if (mse < 1.0e-15)
        {
            return 100.0;
        }
        return 10.0 * std::log10(255.0 * 255.0 / mse);
    }

    cv::Mat gaussianKernel2D(int size, double sigma)
    {
        cv::Mat kernel(size, size, CV_64F);
        const int radius = size / 2;
        double sum = 0.0;
        for (int y = -radius; y <= radius; ++y)
        {
            for (int x = -radius; x <= radius; ++x)
            {
                const double value = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                kernel.at<double>(y + radius, x + radius) = value;
                sum += value;
            }
        }
        kernel /= sum;
        return kernel;
    }

    cv::Mat luma(const cv::Mat& image)
    {
        if (image.channels() == 1)
        {
            cv::Mat gray;
            image.convertTo(gray, CV_64F);
            return gray;
        }

        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (cv::Mat& channel : channels)
        {
            channel.convertTo(channel, CV_64F);
        }

        return 0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2];
    }

    double computeMSSIM(const cv::Mat& original, const cv::Mat& filtered)
    {
        ensureSameShape(original, filtered);

        constexpr int windowSize = 11;
        if (original.cols < windowSize || original.rows < windowSize)
        {
            throw std::runtime_error("Image is too small for MSSIM");
        }

        const cv::Mat originalLuma = luma(original);
        const cv::Mat filteredLuma = luma(filtered);
        const cv::Mat kernel = gaussianKernel2D(windowSize, 1.5);

        const cv::Mat originalSquared = originalLuma.mul(originalLuma);
        const cv::Mat filteredSquared = filteredLuma.mul(filteredLuma);
        const cv::Mat originalFiltered = originalLuma.mul(filteredLuma);

        cv::Mat mu1Full;
        cv::Mat mu2Full;
        cv::Mat sigma11Full;
        cv::Mat sigma22Full;
        cv::Mat sigma12Full;
        cv::filter2D(originalLuma, mu1Full, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(filteredLuma, mu2Full, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(originalSquared, sigma11Full, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(filteredSquared, sigma22Full, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(originalFiltered, sigma12Full, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

        const int radius = windowSize / 2;
        const cv::Rect valid(radius, radius, original.cols - 2 * radius, original.rows - 2 * radius);
        const cv::Mat mu1 = mu1Full(valid);
        const cv::Mat mu2 = mu2Full(valid);

        const cv::Mat mu11 = mu1.mul(mu1);
        const cv::Mat mu22 = mu2.mul(mu2);
        const cv::Mat mu12 = mu1.mul(mu2);
        const cv::Mat sigma11 = sigma11Full(valid) - mu11;
        const cv::Mat sigma22 = sigma22Full(valid) - mu22;
        const cv::Mat sigma12 = sigma12Full(valid) - mu12;

        constexpr double k1 = 0.01;
        constexpr double k2 = 0.03;
        constexpr double l = 255.0;
        constexpr double c1 = (k1 * l) * (k1 * l);
        constexpr double c2 = (k2 * l) * (k2 * l);

        const cv::Mat numerator = (2.0 * mu12 + c1).mul(2.0 * sigma12 + c2);
        const cv::Mat denominator = (mu11 + mu22 + c1).mul(sigma11 + sigma22 + c2);

        cv::Mat ssim;
        cv::divide(numerator, denominator, ssim);
        return cv::mean(ssim)[0];
    }

    size_t nextPowerOfTwo(size_t value)
    {
        size_t result = 1;
        while (result < value)
        {
            result <<= 1;
        }
        return result;
    }

    cv::Mat createFftTransferFunction(int width, int height, double sigma, int centerX, int centerY)
    {
        cv::Mat transfer(height, width, CV_32F);
        for (int y = 0; y < height; ++y)
        {
            const double dy = static_cast<double>(y - centerY);
            float* row = transfer.ptr<float>(y);
            for (int x = 0; x < width; ++x)
            {
                const double dx = static_cast<double>(x - centerX);
                row[x] = static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)));
            }
        }
        return transfer;
    }

    void applyFftCentering(cv::Mat& image)
    {
        for (int y = 0; y < image.rows; ++y)
        {
            float* row = image.ptr<float>(y);
            for (int x = (y % 2 == 0 ? 1 : 0); x < image.cols; x += 2)
            {
                row[x] = -row[x];
            }
        }
    }

    std::vector<cv::Mat> prepareFftChannels(const cv::Mat& input, int paddedWidth, int paddedHeight)
    {
        cv::Mat padded;
        cv::copyMakeBorder(
            input,
            padded,
            0,
            paddedHeight - input.rows,
            0,
            paddedWidth - input.cols,
            cv::BORDER_REPLICATE
        );

        std::vector<cv::Mat> channels;
        cv::split(padded, channels);
        for (cv::Mat& channel : channels)
        {
            channel.convertTo(channel, CV_32F);
            applyFftCentering(channel);
        }
        return channels;
    }

    cv::Mat finishFftChannels(std::vector<cv::Mat>& channels, const cv::Size& originalSize, int outputType)
    {
        std::vector<cv::Mat> croppedChannels;
        croppedChannels.reserve(channels.size());
        const cv::Rect originalRegion(0, 0, originalSize.width, originalSize.height);
        for (cv::Mat& channel : channels)
        {
            applyFftCentering(channel);
            croppedChannels.push_back(channel(originalRegion).clone());
        }

        cv::Mat merged;
        cv::merge(croppedChannels, merged);

        cv::Mat output;
        merged.convertTo(output, outputType);
        return output;
    }

    FilterRun runCpuOperation(const cv::Mat& input, const std::function<void(const cv::Mat&, cv::Mat&)>& operation)
    {
        cv::Mat output;
        operation(input, output);
        return {output, GpuTiming{}};
    }

    FilterRun applyCpuFftGaussianLowpass(const cv::Mat& input, double sigma)
    {
        const int paddedWidth = static_cast<int>(nextPowerOfTwo(2 * static_cast<size_t>(input.cols)));
        const int paddedHeight = static_cast<int>(nextPowerOfTwo(2 * static_cast<size_t>(input.rows)));
        const cv::Mat transfer = createFftTransferFunction(
            paddedWidth,
            paddedHeight,
            sigma,
            paddedWidth / 2,
            paddedHeight / 2
        );

        std::vector<cv::Mat> channels = prepareFftChannels(input, paddedWidth, paddedHeight);
        for (cv::Mat& channel : channels)
        {
            cv::Mat spectrum;
            cv::dft(channel, spectrum, cv::DFT_COMPLEX_OUTPUT);

            std::vector<cv::Mat> planes;
            cv::split(spectrum, planes);
            planes[0] = planes[0].mul(transfer);
            planes[1] = planes[1].mul(transfer);
            cv::merge(planes, spectrum);

            cv::dft(spectrum, channel, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }

        return {finishFftChannels(channels, input.size(), input.type()), GpuTiming{}};
    }

#if defined(IDL_OPENCV_USE_CUDA) && defined(IDL_OPENCV_HAS_CUDA_MODULES)
    int cudaFilterInputType(int inputType)
    {
        const int depth = CV_MAT_DEPTH(inputType);
        const int channels = CV_MAT_CN(inputType);
        if (depth == CV_8U && channels == 3)
        {
            return CV_8UC4;
        }
        return inputType;
    }

    cv::Mat toCudaFilterInput(const cv::Mat& input)
    {
        if (input.type() == CV_8UC3)
        {
            cv::Mat rgba;
            cv::cvtColor(input, rgba, cv::COLOR_RGB2RGBA);
            return rgba;
        }
        return input;
    }

    cv::Mat fromCudaFilterOutput(const cv::Mat& output, int originalType)
    {
        if (originalType == CV_8UC3 && output.type() == CV_8UC4)
        {
            cv::Mat rgb;
            cv::cvtColor(output, rgb, cv::COLOR_RGBA2RGB);
            return rgb;
        }
        return output;
    }

    FilterRun runCudaFilter(const cv::Mat& input, const cv::Ptr<cv::cuda::Filter>& filter)
    {
        const cv::Mat uploadInput = toCudaFilterInput(input);
        cv::cuda::Stream stream;
        cv::cuda::GpuMat gpuInput;
        cv::cuda::GpuMat gpuOutput;
        cv::Mat output;
        GpuTiming timing;
        timing.available = true;

        auto start = Clock::now();
        gpuInput.upload(uploadInput, stream);
        stream.waitForCompletion();
        timing.hostToDeviceMs = elapsedMs(start, Clock::now());

        start = Clock::now();
        filter->apply(gpuInput, gpuOutput, stream);
        stream.waitForCompletion();
        timing.kernelMs = elapsedMs(start, Clock::now());

        start = Clock::now();
        gpuOutput.download(output, stream);
        stream.waitForCompletion();
        timing.deviceToHostMs = elapsedMs(start, Clock::now());
        timing.deviceTotalMs = timing.hostToDeviceMs + timing.kernelMs + timing.deviceToHostMs;

        return {fromCudaFilterOutput(output, input.type()), timing};
    }

    FilterRun applyCudaBilateral(const cv::Mat& input, int diameter, double sigmaColor, double sigmaSpace)
    {
        cv::cuda::Stream stream;
        cv::cuda::GpuMat gpuInput;
        cv::cuda::GpuMat gpuOutput;
        cv::Mat output;
        GpuTiming timing;
        timing.available = true;

        auto start = Clock::now();
        gpuInput.upload(input, stream);
        stream.waitForCompletion();
        timing.hostToDeviceMs = elapsedMs(start, Clock::now());

        start = Clock::now();
        cv::cuda::bilateralFilter(
            gpuInput,
            gpuOutput,
            diameter,
            static_cast<float>(sigmaColor),
            static_cast<float>(sigmaSpace),
            cv::BORDER_REPLICATE,
            stream
        );
        stream.waitForCompletion();
        timing.kernelMs = elapsedMs(start, Clock::now());

        start = Clock::now();
        gpuOutput.download(output, stream);
        stream.waitForCompletion();
        timing.deviceToHostMs = elapsedMs(start, Clock::now());
        timing.deviceTotalMs = timing.hostToDeviceMs + timing.kernelMs + timing.deviceToHostMs;

        return {output, timing};
    }

    FilterRun applyCudaFftGaussianLowpass(const cv::Mat& input, double sigma)
    {
        const int paddedWidth = static_cast<int>(nextPowerOfTwo(2 * static_cast<size_t>(input.cols)));
        const int paddedHeight = static_cast<int>(nextPowerOfTwo(2 * static_cast<size_t>(input.rows)));
        const int packedSpectrumWidth = paddedWidth / 2 + 1;
        const cv::Mat transfer = createFftTransferFunction(
            packedSpectrumWidth,
            paddedHeight,
            sigma,
            paddedWidth / 2,
            paddedHeight / 2
        );

        std::vector<cv::Mat> transferPlanes = {transfer, transfer};
        cv::Mat transferComplex;
        cv::merge(transferPlanes, transferComplex);

        std::vector<cv::Mat> channels = prepareFftChannels(input, paddedWidth, paddedHeight);

        cv::cuda::Stream stream;
        cv::cuda::GpuMat gpuTransfer;
        GpuTiming timing;
        timing.available = true;

        auto start = Clock::now();
        gpuTransfer.upload(transferComplex, stream);
        stream.waitForCompletion();
        timing.hostToDeviceMs += elapsedMs(start, Clock::now());

        for (cv::Mat& channel : channels)
        {
            cv::cuda::GpuMat gpuInput;
            cv::cuda::GpuMat gpuSpectrum;
            cv::cuda::GpuMat gpuFilteredSpectrum;
            cv::cuda::GpuMat gpuOutput;

            start = Clock::now();
            gpuInput.upload(channel, stream);
            stream.waitForCompletion();
            timing.hostToDeviceMs += elapsedMs(start, Clock::now());

            start = Clock::now();
            cv::cuda::dft(gpuInput, gpuSpectrum, cv::Size(paddedWidth, paddedHeight), 0, stream);
            cv::cuda::multiply(gpuSpectrum, gpuTransfer, gpuFilteredSpectrum, 1.0, -1, stream);
            cv::cuda::dft(
                gpuFilteredSpectrum,
                gpuOutput,
                cv::Size(paddedWidth, paddedHeight),
                cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE,
                stream
            );
            stream.waitForCompletion();
            timing.kernelMs += elapsedMs(start, Clock::now());

            start = Clock::now();
            gpuOutput.download(channel, stream);
            stream.waitForCompletion();
            timing.deviceToHostMs += elapsedMs(start, Clock::now());
        }

        timing.deviceTotalMs = timing.hostToDeviceMs + timing.kernelMs + timing.deviceToHostMs;
        return {finishFftChannels(channels, input.size(), input.type()), timing};
    }
#endif

    int bilateralDiameter(double sigmaDomain)
    {
        return 2 * static_cast<int>(std::ceil(3.0 * sigmaDomain)) + 1;
    }

    std::vector<FilterScenario> createFilterScenarios(int inputType)
    {
#if defined(IDL_OPENCV_USE_CUDA) && defined(IDL_OPENCV_HAS_CUDA_MODULES)
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        {
            throw std::runtime_error("No CUDA-enabled OpenCV device found");
        }

        cv::cuda::setDevice(0);
        static bool warnedAboutMedian = false;
        if (!warnedAboutMedian)
        {
            std::cerr << "Warning: OpenCV CUDA does not provide a median filter; median scenarios are skipped.\n";
            warnedAboutMedian = true;
        }

        const int cudaInputType = cudaFilterInputType(inputType);
        const cv::Ptr<cv::cuda::Filter> meanRadius1 =
            cv::cuda::createBoxFilter(cudaInputType, cudaInputType, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
        const cv::Ptr<cv::cuda::Filter> meanRadius5 =
            cv::cuda::createBoxFilter(cudaInputType, cudaInputType, cv::Size(11, 11), cv::Point(-1, -1), cv::BORDER_REPLICATE);
        const cv::Ptr<cv::cuda::Filter> gaussianSigma1 =
            cv::cuda::createGaussianFilter(cudaInputType, cudaInputType, cv::Size(7, 7), 1.0, 1.0, cv::BORDER_REPLICATE);
        const cv::Ptr<cv::cuda::Filter> gaussianSigma3 =
            cv::cuda::createGaussianFilter(cudaInputType, cudaInputType, cv::Size(19, 19), 3.0, 3.0, cv::BORDER_REPLICATE);

        return {
            {"mean", "opencv_mean_radius_1x1", [meanRadius1](const cv::Mat& input) {
                 return runCudaFilter(input, meanRadius1);
             }},
            {"mean", "opencv_mean_radius_5x5", [meanRadius5](const cv::Mat& input) {
                 return runCudaFilter(input, meanRadius5);
             }},
            {"gaussian", "opencv_gaussian_sigma_1", [gaussianSigma1](const cv::Mat& input) {
                 return runCudaFilter(input, gaussianSigma1);
             }},
            {"gaussian", "opencv_gaussian_sigma_3", [gaussianSigma3](const cv::Mat& input) {
                 return runCudaFilter(input, gaussianSigma3);
             }},
            {"bilateral", "opencv_bilateral_domain_1_range_20", [](const cv::Mat& input) {
                 return applyCudaBilateral(input, bilateralDiameter(1.0), 20.0, 1.0);
             }},
            {"bilateral", "opencv_bilateral_domain_3_range_50", [](const cv::Mat& input) {
                 return applyCudaBilateral(input, bilateralDiameter(3.0), 50.0, 3.0);
             }},
            {"fft", "opencv_fft_sigma_10", [](const cv::Mat& input) {
                 return applyCudaFftGaussianLowpass(input, 10.0);
             }},
            {"fft", "opencv_fft_sigma_50", [](const cv::Mat& input) {
                 return applyCudaFftGaussianLowpass(input, 50.0);
             }}
        };
#else
        (void)inputType;
        return {
            {"mean", "opencv_mean_radius_1x1", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::blur(src, dst, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
                 });
             }},
            {"mean", "opencv_mean_radius_5x5", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::blur(src, dst, cv::Size(11, 11), cv::Point(-1, -1), cv::BORDER_REPLICATE);
                 });
             }},
            {"gaussian", "opencv_gaussian_sigma_1", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::GaussianBlur(src, dst, cv::Size(7, 7), 1.0, 1.0, cv::BORDER_REPLICATE);
                 });
             }},
            {"gaussian", "opencv_gaussian_sigma_3", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::GaussianBlur(src, dst, cv::Size(19, 19), 3.0, 3.0, cv::BORDER_REPLICATE);
                 });
             }},
            {"median", "opencv_median_radius_1x1", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::medianBlur(src, dst, 3);
                 });
             }},
            {"median", "opencv_median_radius_5x5", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::medianBlur(src, dst, 11);
                 });
             }},
            {"bilateral", "opencv_bilateral_domain_1_range_20", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::bilateralFilter(src, dst, bilateralDiameter(1.0), 20.0, 1.0, cv::BORDER_REPLICATE);
                 });
             }},
            {"bilateral", "opencv_bilateral_domain_3_range_50", [](const cv::Mat& input) {
                 return runCpuOperation(input, [](const cv::Mat& src, cv::Mat& dst) {
                     cv::bilateralFilter(src, dst, bilateralDiameter(3.0), 50.0, 3.0, cv::BORDER_REPLICATE);
                 });
             }},
            {"fft", "opencv_fft_sigma_10", [](const cv::Mat& input) {
                 return applyCpuFftGaussianLowpass(input, 10.0);
             }},
            {"fft", "opencv_fft_sigma_50", [](const cv::Mat& input) {
                 return applyCpuFftGaussianLowpass(input, 50.0);
             }}
        };
#endif
    }

    BenchmarkResult runBenchmark(const FilterScenario& filterScenario, const cv::Mat& input,
                                 const cv::Mat& original, uint32_t repetitions)
    {
        if (repetitions == 0)
        {
            throw std::invalid_argument("Benchmark repetitions must be positive");
        }

        cv::Mat output;
        GpuTiming totalGpuTiming;
        const double imageMegapixels = static_cast<double>(input.cols) * static_cast<double>(input.rows) / 1.0e6;

        const auto start = Clock::now();
        for (uint32_t iteration = 0; iteration < repetitions; ++iteration)
        {
            const FilterRun run = filterScenario.apply(input);
            output = run.output;
            if (run.gpuTiming.available)
            {
                totalGpuTiming.available = true;
                totalGpuTiming.hostToDeviceMs += run.gpuTiming.hostToDeviceMs;
                totalGpuTiming.kernelMs += run.gpuTiming.kernelMs;
                totalGpuTiming.deviceToHostMs += run.gpuTiming.deviceToHostMs;
                totalGpuTiming.deviceTotalMs += run.gpuTiming.deviceTotalMs;
            }
        }
        const auto end = Clock::now();

        BenchmarkResult result;
        result.executionTimeMs = elapsedMs(start, end) / static_cast<double>(repetitions);
        result.throughputMPs = result.executionTimeMs > 0.0
            ? imageMegapixels / result.executionTimeMs * 1000.0
            : 0.0;
        result.mse = computeMSE(original, output);
        result.psnr = computePSNR(original, output);
        result.mssim = computeMSSIM(original, output);

        if (totalGpuTiming.available)
        {
            result.gpuTiming.available = true;
            result.gpuTiming.hostToDeviceMs = totalGpuTiming.hostToDeviceMs / repetitions;
            result.gpuTiming.kernelMs = totalGpuTiming.kernelMs / repetitions;
            result.gpuTiming.deviceToHostMs = totalGpuTiming.deviceToHostMs / repetitions;
            result.gpuTiming.deviceTotalMs = totalGpuTiming.deviceTotalMs / repetitions;
        }

        return result;
    }

    void writeResult(std::ofstream& output, const std::string& noiseName,
                     const std::string& filterName, const BenchmarkResult& result)
    {
        output << IDL_OPENCV_BENCHMARK_BACKEND << ','
               << noiseName << ','
               << filterName << ','
               << std::fixed << std::setprecision(6)
               << result.executionTimeMs << ','
               << result.throughputMPs << ','
               << result.gpuTiming.hostToDeviceMs << ','
               << result.gpuTiming.kernelMs << ','
               << result.gpuTiming.deviceToHostMs << ','
               << result.gpuTiming.deviceTotalMs << ','
               << result.mse << ','
               << result.psnr << ','
               << result.mssim << '\n';
        output.flush();
    }
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3 || argc > 5)
        {
            std::cerr << "Usage: " << argv[0]
                      << " <clean-image> <results.csv> [repetitions] [all|mean|gaussian|median|bilateral|fft]\n";
            return 1;
        }

        const uint32_t repetitions = argc >= 4 ? parseRepetitions(argv[3]) : 10;
        const std::string requestedGroup = argc == 5 ? argv[4] : "all";
        if (requestedGroup != "all" && requestedGroup != "mean" && requestedGroup != "gaussian" &&
            requestedGroup != "median" && requestedGroup != "bilateral" && requestedGroup != "fft")
        {
            std::cerr << "Unknown filter group: " << requestedGroup << '\n';
            return 1;
        }

        const cv::Mat original = loadImage(argv[1]);
        std::ofstream output(argv[2]);
        if (!output)
        {
            std::cerr << "Cannot open CSV output file: " << argv[2] << '\n';
            return 1;
        }

        output << "backend,noise,filter,execution_time_ms,throughput_mps,h2d_ms,kernel_ms,d2h_ms,device_total_ms,mse,psnr,mssim\n";

        const std::vector<NoiseScenario> noises = {
            {"poisson_05", [](const cv::Mat& input) {
                 return applyPoissonNoise(input, 0.5, 1403);
             }},
            {"saltnpepper_01_01", [](const cv::Mat& input) {
                 return applySaltAndPepperNoise(input, 0.1, 0.1, 1403);
             }},
            {"gaussian_50", [](const cv::Mat& input) {
                 return applyGaussianNoise(input, 0.0, 50.0, 1403);
             }}
        };

        for (const NoiseScenario& noiseScenario : noises)
        {
            const cv::Mat noisy = noiseScenario.apply(original);
            const std::vector<FilterScenario> filters = createFilterScenarios(noisy.type());

            for (const FilterScenario& filterScenario : filters)
            {
                if (requestedGroup != "all" && requestedGroup != filterScenario.group)
                {
                    continue;
                }

                const BenchmarkResult result = runBenchmark(filterScenario, noisy, original, repetitions);
                writeResult(output, noiseScenario.name, filterScenario.name, result);
                std::cout << IDL_OPENCV_BENCHMARK_BACKEND << " | " << noiseScenario.name << " | "
                          << filterScenario.name << " | " << result.executionTimeMs << " ms\n";
            }
        }
    }
    catch (const cv::Exception& error)
    {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
