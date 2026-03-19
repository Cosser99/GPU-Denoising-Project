#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK 16

//---------------------------------------------
// Kernel gaussiano
//---------------------------------------------
void createGaussianKernel(float* kernel, int k, float sigma)
{
    int half = k / 2;
    float sum = 0.0f;

    for (int i = -half; i <= half; i++)
    {
        float value = expf(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + half] = value;
        sum += value;
    }

    for (int i = 0; i < k; i++)
        kernel[i] /= sum;
}

//---------------------------------------------
// Convoluzione orizzontale
//---------------------------------------------
__global__
void gaussianHorizontal(float* input, float* output,
                        float* kernel,
                        int width, int height, int k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = k / 2;
    float sum = 0.0f;

    for (int i = -half; i <= half; i++)
    {
        int xi = min(max(x + i, 0), width - 1);
        sum += input[y * width + xi] * kernel[i + half];
    }

    output[y * width + x] = sum;
}

//---------------------------------------------
// Convoluzione verticale
//---------------------------------------------
__global__
void gaussianVertical(float* input, float* output,
                      float* kernel,
                      int width, int height, int k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = k / 2;
    float sum = 0.0f;

    for (int i = -half; i <= half; i++)
    {
        int yi = min(max(y + i, 0), height - 1);
        sum += input[yi * width + x] * kernel[i + half];
    }

    output[y * width + x] = sum;
}

//---------------------------------------------
// MAIN
//---------------------------------------------
int main(int argc,char *argv[])
{
    printf("Gaussian Blur CUDA\n");

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if(img.empty())
    {
        printf("Errore caricamento immagine\n");
        return -1;
    }

    cv::imshow("Original", img);
    cv::waitKey(0);

    int width = img.cols;
    int height = img.rows;

    int k = 7;
    float sigma = 1.5f;

    int pixels = width * height;

    //---------------------------------------------
    // Convertiamo immagine in float
    //---------------------------------------------

    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);

    //---------------------------------------------
    // Kernel gaussiano
    //---------------------------------------------

    float* h_kernel = new float[k];
    createGaussianKernel(h_kernel, k, sigma);

    //---------------------------------------------
    // GPU memory
    //---------------------------------------------

    float *d_input, *d_temp, *d_output, *d_kernel;

    cudaMalloc(&d_input, pixels * sizeof(float));
    cudaMalloc(&d_temp, pixels * sizeof(float));
    cudaMalloc(&d_output, pixels * sizeof(float));
    cudaMalloc(&d_kernel, k * sizeof(float));

    //---------------------------------------------
    // Copia su GPU
    //---------------------------------------------

    cudaMemcpy(d_input,
               imgFloat.ptr<float>(),
               pixels * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_kernel,
               h_kernel,
               k * sizeof(float),
               cudaMemcpyHostToDevice);

    //---------------------------------------------
    // Grid config
    //---------------------------------------------

    dim3 block(BLOCK, BLOCK);
    dim3 grid((width + BLOCK - 1) / BLOCK,
              (height + BLOCK - 1) / BLOCK);

    //---------------------------------------------
    // Kernel launch
    //---------------------------------------------
    printf("Processing image");
    for(int passi=20;passi>0;passi--)
    {
    gaussianHorizontal<<<grid, block>>>(
        d_input, d_temp, d_kernel,
        width, height, k);

    gaussianVertical<<<grid, block>>>(
        d_temp, d_output, d_kernel,
        width, height, k);
    cudaDeviceSynchronize();
    cudaMemcpy(d_input,d_output,pixels*sizeof(float),cudaMemcpyDeviceToDevice);
    }
    //---------------------------------------------
    // Copia risultato
    //---------------------------------------------

    cv::Mat resultFloat(height, width, CV_32F);

    cudaMemcpy(resultFloat.ptr<float>(),
               d_output,
               pixels * sizeof(float),
               cudaMemcpyDeviceToHost);

    //---------------------------------------------
    // Convertiamo in uchar per salvare
    //---------------------------------------------

    cv::Mat result;
    resultFloat.convertTo(result, CV_8U);

    //---------------------------------------------
    // Output
    //---------------------------------------------
    printf("End Processing");
    cv::imshow("Gaussian", result);
    cv::imwrite("Gaussian.png", result);

    cv::waitKey(0);

    //---------------------------------------------
    // Free memoria
    //---------------------------------------------

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);

    delete[] h_kernel;

    return 0;
}
