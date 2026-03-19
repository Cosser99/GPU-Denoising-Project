// test program

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int* a, int* b, int* c)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main()
{
    int count;
    printf("Sono qui!");
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("Sono dopo getdevice");
    if (err != cudaSuccess) {
        printf("Errore CUDA: %s\n", cudaGetErrorString(err));
        printf("dopo");
        return 1;
    }

    printf("Numero di GPU CUDA rilevate: %d\n", count);
    return 0;
}
