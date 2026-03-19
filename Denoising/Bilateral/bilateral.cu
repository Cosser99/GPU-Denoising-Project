#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 16

__device__ float gaussian(float x, float sigma)
{
    return expf(-(x * x) / (2 * sigma * sigma));
}

__global__ void bilateralFilterKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int radius,
    float sigma_spatial,
    float sigma_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < channels; c++)
    {
        float sum = 0.0f;
        float norm = 0.0f;

        float center = input[(y * width + x) * channels + c];

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && ny >= 0 && nx < width && ny < height)
                {
                    float neighbor = input[(ny * width + nx) * channels + c];

                    float spatial = gaussian(sqrtf(dx * dx + dy * dy), sigma_spatial);
                    float range = gaussian(neighbor - center, sigma_range);

                    float weight = spatial * range;

                    sum += neighbor * weight;
                    norm += weight;
                }
            }
        }

        output[(y * width + x) * channels + c] = (unsigned char)(sum / norm);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: ./bilateral_cuda image.jpg\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1]);

    if (img.empty())
    {
        std::cout << "Errore caricamento immagine\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    size_t bytes = width * height * channels * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, img.data, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int radius = 3;
    float sigma_spatial = 3.0f;
    float sigma_range = 25.0f;

    bilateralFilterKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height,
        channels,
        radius,
        sigma_spatial,
        sigma_range);

    cv::Mat result(height, width, img.type());

    cudaMemcpy(result.data, d_output, bytes, cudaMemcpyDeviceToHost);

    cv::imwrite("bilateral.png", result);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Denoising completato -> bilateral.png\n";

    return 0;
}