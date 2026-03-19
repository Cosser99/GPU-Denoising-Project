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

// Kernel Salt & Pepper
__global__ void addSaltPepperNoise(unsigned char *img, curandState *state,
                                   int width, int height, float saltProb, float pepperProb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curandState localState = state[idx];

    float r = curand_uniform(&localState); // 0-1
    if (r < saltProb) {
        img[idx] = 255; // "Sale"
    } else if (r < saltProb + pepperProb) {
        img[idx] = 0;   // "Pepe"
    }
    state[idx] = localState;
}

int main()
{
    // Carica immagine
    cv::Mat image = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
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
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    initRand<<<grid, block>>>(d_state, time(NULL), width, height);

    addSaltPepperNoise<<<grid, block>>>(d_img, d_state, width, height,
                                        0.05f, 0.05f); // 5% sale, 5% pepe

    cudaMemcpy(image.data, d_img, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("spout.png", image);

    cudaFree(d_img);
    cudaFree(d_state);
    std::cout << "Output salvato in spout.png\n";
    return 0;
}