#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846

struct GPUComplex {
    float r;
    float i;
};

__device__ GPUComplex cadd(GPUComplex a, GPUComplex b){
    GPUComplex c;
    c.r = a.r + b.r;
    c.i = a.i + b.i;
    return c;
}

__device__ GPUComplex cmul(GPUComplex a, GPUComplex b){
    GPUComplex c;
    c.r = a.r*b.r - a.i*b.i;
    c.i = a.r*b.i + a.i*b.r;
    return c;
}

__device__ GPUComplex cexp(float theta){
    GPUComplex c;
    c.r = cosf(theta);
    c.i = sinf(theta);
    return c;
}

__global__ void gaussianFilter(GPUComplex* freq, int width, int height, float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    int cx = width/2;
    int cy = height/2;

    float dx = x - cx;
    float dy = y - cy;

    float d2 = dx*dx + dy*dy;

    float g = expf(-d2/(2*sigma*sigma));

    int idx = y*width + x;

    freq[idx].r *= g;
    freq[idx].i *= g;
}

__global__ void dft2D(float* input, GPUComplex* output, int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if(u >= width || v >= height) return;

    GPUComplex sum;
    sum.r = 0;
    sum.i = 0;

    for(int x=0;x<width;x++){
        for(int y=0;y<height;y++){

            float angle = -2.0f*PI*((u*x/(float)width) + (v*y/(float)height));

            GPUComplex e = cexp(angle);

            float val = input[y*width + x];

            sum.r += val * e.r;
            sum.i += val * e.i;
        }
    }

    output[v*width + u] = sum;
}

__global__ void idft2D(GPUComplex* input, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    GPUComplex sum;
    sum.r = 0;
    sum.i = 0;

    for(int u=0;u<width;u++){
        for(int v=0;v<height;v++){

            float angle = 2.0f*PI*((u*x/(float)width) + (v*y/(float)height));

            GPUComplex e = cexp(angle);

            GPUComplex F = input[v*width + u];

            sum.r += F.r * e.r - F.i * e.i;
            sum.i += F.r * e.i + F.i * e.r;
        }
    }

    output[y*width + x] = sum.r / (width*height);
}

int main(int argc,char** argv)
{
    printf("Init program");
    Mat img = imread("input.png", IMREAD_GRAYSCALE);

    int w = img.cols;
    int h = img.rows;

    Mat imgf;
    img.convertTo(imgf, CV_32F);

    float* d_input;
    float* d_output;
    GPUComplex* d_freq;

    cudaMalloc(&d_input, w*h*sizeof(float));
    cudaMalloc(&d_output, w*h*sizeof(float));
    cudaMalloc(&d_freq, w*h*sizeof(GPUComplex));

    cudaMemcpy(d_input, imgf.ptr<float>(), w*h*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((w+15)/16,(h+15)/16);

    dft2D<<<grid,block>>>(d_input,d_freq,w,h);

    cudaDeviceSynchronize();

    gaussianFilter<<<grid,block>>>(d_freq,w,h,50.0f);

    cudaDeviceSynchronize();

    idft2D<<<grid,block>>>(d_freq,d_output,w,h);

    cudaDeviceSynchronize();

    Mat result(h,w,CV_32F);

    cudaMemcpy(result.ptr<float>(), d_output, w*h*sizeof(float), cudaMemcpyDeviceToHost);

    Mat finalImg;
    normalize(result,finalImg,0,255,NORM_MINMAX);
    finalImg.convertTo(finalImg,CV_8U);

    imwrite("freq.png",finalImg);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_freq);

    return 0;
}