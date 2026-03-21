#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define BLOCK_SIZE 16

// inizializza generatori casuali
__global__ void initRand(curandState *state, unsigned long seed, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int id = y * width + x;
    curand_init(seed, id, 0, &state[id]);
}

// kernel per rumore gaussiano
__global__ void addGaussianNoiseGray(unsigned char *img,
                                     curandState *state,
                                     int width,
                                     int height,
                                     float mean,
                                     float stddev)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    curandState localState = state[idx];

    float noise = curand_normal(&localState) * stddev + mean;

    int value = img[idx] + noise;

    value = max(0, min(255, value));

    img[idx] = (unsigned char)value;

    state[idx] = localState;
}

int main(int argc,char *argv[])
{
   
    if(argc<2) 
    {
        printf("ERRORE : argument must be <input image.png>");
        return;
    }
    char path[64];
    snprintf(path,sizeof(path),"..\\..\\image\\%s",argv[1]);
    // carica immagine
    cv::Mat image = cv::imread(path);

    if (image.empty())
    {
        std::cout << "Errore caricamento immagine\n";
        return -1;
    }

    // conversione grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;

    size_t imgSize = width * height * sizeof(unsigned char);

    unsigned char *d_img;
    curandState *d_state;

    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_state, width * height * sizeof(curandState));

    cudaMemcpy(d_img, gray.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    initRand<<<grid, block>>>(d_state, time(NULL), width, height);

    addGaussianNoiseGray<<<grid, block>>>(
        d_img,
        d_state,
        width,
        height,
        0.0f,   // media
        20.0f   // deviazione standard
    );

    cudaMemcpy(gray.data, d_img, imgSize, cudaMemcpyDeviceToHost);

    cv::imwrite("output.png", gray);

    cudaFree(d_img);
    cudaFree(d_state);

    std::cout << "Output salvato in output.png\n";

    return 0;
}