#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;

void createGaussianFilter(Size size, Mat& filter, float sigma)
{
    filter.create(size, CV_32FC2);

    int cx = size.width / 2;
    int cy = size.height / 2;

    for(int y=0; y<size.height; y++)
    {
        for(int x=0; x<size.width; x++)
        {
            float dx = x - cx;
            float dy = y - cy;
            float d2 = dx*dx + dy*dy;

            float g = exp(-d2/(2*sigma*sigma));

            filter.at<Vec2f>(y,x)[0] = g;
            filter.at<Vec2f>(y,x)[1] = g;
        }
    }
}

int main()
{
    Mat img = imread("input.png", IMREAD_GRAYSCALE);

    Mat floatImg;
    img.convertTo(floatImg, CV_32F);

    cuda::GpuMat d_img(floatImg);

    // crea piano complesso
    cuda::GpuMat planes[] = {d_img, cuda::GpuMat::zeros(d_img.size(), CV_32F)};
    cuda::GpuMat complexImg;

    cuda::merge(planes,2,complexImg);

    // FFT
    cuda::GpuMat dftImg;
    cuda::dft(complexImg, dftImg);

    // filtro
    Mat filter;
    createGaussianFilter(img.size(), filter, 50);

    cuda::GpuMat d_filter(filter);

    // applica filtro
    cuda::GpuMat filtered;
    cuda::mulSpectrums(dftImg, d_filter, filtered, 0);

    // IFFT
    cuda::GpuMat invDFT;
    cuda::dft(filtered, invDFT, Size(), DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

    Mat result;
    invDFT.download(result);

    result.convertTo(result, CV_8U);

    imshow("denoised", result);
    waitKey(0);

    return 0;
}