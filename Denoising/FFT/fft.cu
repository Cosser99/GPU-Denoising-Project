#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void gaussianFilter(cufftComplex* data,int w,int h,float sigma)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x>=w || y>=h) return;

    int idx = y*w + x;

    float cx = w/2.0f;
    float cy = h/2.0f;

    float dx = x - cx;
    float dy = y - cy;

    float d2 = dx*dx + dy*dy;

    float g = expf(-d2/(2*sigma*sigma));

    data[idx].x *= g;
    data[idx].y *= g;
}

__global__ void fftShift(cufftComplex* data,int w,int h)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x>=w/2 || y>=h/2) return;

    int x2 = x + w/2;
    int y2 = y + h/2;

    int i1 = y*w + x;
    int i2 = y2*w + x2;

    cufftComplex tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
}
// Compiler : compfft.bat
int main()
{
    std::cout<<"FFT Denoiser\n";

    cv::Mat img = cv::imread("input.png",cv::IMREAD_GRAYSCALE);

    if(img.empty())
    {
        std::cout<<"Errore caricamento immagine\n";
        return -1;
    }

    img.convertTo(img,CV_32F);

    int width = img.cols;
    int height = img.rows;

    cufftComplex* d_data;

    cudaMalloc(&d_data,sizeof(cufftComplex)*width*height);

    std::vector<cufftComplex> host(width*height);

    for(int y=0;y<height;y++)
    for(int x=0;x<width;x++)
    {
        int i = y*width + x;

        host[i].x = img.at<float>(y,x);
        host[i].y = 0.0f;
    }

    cudaMemcpy(d_data,host.data(),
               sizeof(cufftComplex)*width*height,
               cudaMemcpyHostToDevice);

    cufftHandle plan;

    cufftPlan2d(&plan,height,width,CUFFT_C2C);

    cufftExecC2C(plan,d_data,d_data,CUFFT_FORWARD);

    dim3 block(16,16);
    dim3 grid((width+15)/16,(height+15)/16);

    fftShift<<<grid,block>>>(d_data,width,height);
    cudaDeviceSynchronize();

    gaussianFilter<<<grid,block>>>(d_data,width,height,100.0f);
    cudaDeviceSynchronize();

    fftShift<<<grid,block>>>(d_data,width,height);
    cudaDeviceSynchronize();

    cufftExecC2C(plan,d_data,d_data,CUFFT_INVERSE);

    cudaMemcpy(host.data(),d_data,
               sizeof(cufftComplex)*width*height,
               cudaMemcpyDeviceToHost);

    cv::Mat result(height,width,CV_32F);

    for(int y=0;y<height;y++)
    for(int x=0;x<width;x++)
    {
        int i = y*width+x;

        result.at<float>(y,x) = host[i].x/(width*height);
    }

    cv::normalize(result,result,0,255,cv::NORM_MINMAX);

    result.convertTo(result,CV_8U);

    cv::imwrite("fft_denoised.png",result);

    cufftDestroy(plan);
    cudaFree(d_data);

    std::cout<<"Denoising completato\n";

    return 0;
}