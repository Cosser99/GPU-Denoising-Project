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
    printf("Init program\n");
    fflush(stdout);
    // inizializzazione
    for (int i = 0; i < 256; i++)
    {
        vector_a[i] = 1;
        vector_b[i] = 1;
    }
    printf("Sono dopo il for di inizializzazione vettore");
    // lancio kernel
    add<<<1, 256>>>(vector_a, vector_b, vector_c);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    
    printf("Sono dopo il kernel");
    // sincronizzazione + controllo errori
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Sono dopo il controllo di cudageterrorrstring");
    // somma dei risultati
    int result_sum = 0;
    for (int i = 0; i < 256; i++)
    {
        result_sum += vector_c[i];
    }

    printf("Result: %d\n", result_sum);

    return 0;
}
