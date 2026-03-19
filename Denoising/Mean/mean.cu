//mean_filter
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Media finestra di 9 valori
__device__ unsigned char mean9(unsigned char *v) {
    int sum = 0;
    for(int i = 0; i < 9; i++)
        sum += v[i];

    return sum / 9;
}

__device__ unsigned char mean25(unsigned char *v)
{
    int sum=0;
    for(int i=0;i<25;i++)
    {
        sum+=v[i];
    }
    return sum/25;

}

__device__ unsigned char mean(unsigned char *v,int size)
{
    int sum=0;
    for(int i=0;i<size;i++)
    {
        sum+=v[i];
    }
    return sum/size;

}

// Kernel median filter 3x3
__global__ void meanKernel(unsigned char *in, unsigned char *out, int width, int height, size_t step,int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    //unsigned char win[9];
    //unsigned char win[25];
    unsigned char win[120];
    int k = 0;
    for(int j=-size/2; j<=size/2; j++)
        for(int i=-size/2; i<=size/2; i++)
            win[k++] = in[(y+j)*step + (x+i)];

    out[y*step + x] = mean(win,size);
}

int main(int argc,char *argv[])
{
    int size;
    printf("Argomento : %s",argv[1]);
    printf("Inserisci size: ");
    scanf("%d",&size);
       cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if(img.empty()) { 
        std::cerr << "Errore: immagine non trovata!" << std::endl;
        return -1;
    }

    // Assicuriamoci che l'immagine sia continua
    if(!img.isContinuous()) img = img.clone();
    cv::imshow("Original",img);
        int width = img.cols;
    int height = img.rows;
    size_t step = img.step; // passo reale (stride)
    printf("Step : %zu\n",step);   //debug
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, height*step);
    cudaMalloc(&d_out, height*step);
    cudaMemcpy(d_in, img.data, height*step, cudaMemcpyHostToDevice);
    
    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    
    cv::Mat out(height, width, CV_8UC1);

    
        meanKernel<<<grid, block>>>(d_in, d_out, width, height, step,size);
        cudaDeviceSynchronize();


    cudaMemcpy(out.data, d_out, height*step, cudaMemcpyDeviceToHost);
    char str[32];
    sprintf(str,"mean%d.png",size);
    cv::imwrite(str, out);
    cv::imshow("Output",out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    cudaFree(d_in);
    cudaFree(d_out);
}