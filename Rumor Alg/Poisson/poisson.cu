#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define BLOCK_SIZE 16

// Inizializza generatori casuali
__global__ void initRand(curandState *state, unsigned long seed, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = y * width + x;
    curand_init(seed, id, 0, &state[id]);
}

// Kernel Poisson noise
__global__ void addPoissonNoise(unsigned char *img, curandState *state,
                               int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curandState localState = state[idx];

    float lambda = img[idx]; // lambda proporzionale al valore del pixel (0-255)
    float L = expf(-lambda);
    float p = 1.0f;
    int k = 0;

    do {
        k++;
        float u = curand_uniform(&localState);
        p *= u;
    } while (p > L);

    int value = k - 1;
    value = max(0, min(255, value));
    img[idx] = (unsigned char)value;

    state[idx] = localState;
}

int main()
{
    cv::Mat image = cv::imread("input_poisson.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) { std::cout << "Errore caricamento immagine\n"; return -1; }

    int width = image.cols;
    int height = image.rows;
    size_t imgSize = width * height * sizeof(unsigned char);

    unsigned char *d_img;
    curandState *d_state;
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_state, width * height * sizeof(curandState));
    cudaMemcpy(d_img, image.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1)/BLOCK_SIZE,
              (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    initRand<<<grid, block>>>(d_state, time(NULL), width, height);

    addPoissonNoise<<<grid, block>>>(d_img, d_state, width, height);

    cudaMemcpy(image.data, d_img, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("pout.png", image);

    cudaFree(d_img);
    cudaFree(d_state);

    std::cout << "Output salvato in pout.png\n";
    return 0;
}