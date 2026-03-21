#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


// Ordina 9 valori e ritorna la mediana
__device__ unsigned char median9(unsigned char *v) {
    for(int i=0;i<9;i++)
        for(int j=i+1;j<9;j++)
            if(v[j] < v[i]) {
                unsigned char tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
    return v[4];
}

// Kernel median filter 3x3
__global__ void medianKernel(unsigned char *in, unsigned char *out, int width, int height, size_t step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    unsigned char win[9];
    int k = 0;
    for(int j=-1; j<=1; j++)
        for(int i=-1; i<=1; i++)
            win[k++] = in[(y+j)*step + (x+i)];

    out[y*step + x] = median9(win);
}



int main(int argc,char *argv[]) {

    if(argc<2) 
    {
        printf("ERRORE : argument must be <input image.png>");
        return;
    }
    char path[64];
    snprintf(path,sizeof(path),"..\\..\\image\\%s",argv[1]);
    // carica immagine
    cv::Mat img = cv::imread(path,cv::IMREAD_GRAYSCALE);




    if(img.empty()) { 
        std::cerr << "Errore: immagine non trovata!" << std::endl;
        return -1;
    }
    //show image
    cv::imshow("Original",img);
    cv::waitKey(0);
    // Assicuriamoci che l'immagine sia continua
    if(!img.isContinuous()) img = img.clone();

    int width = img.cols;
    int height = img.rows;
    size_t step = img.step; // passo reale (stride)

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, height*step);
    cudaMalloc(&d_out, height*step);
    cudaMemcpy(d_in, img.data, height*step, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    
    cv::Mat out(height, width, CV_8UC1);

    
        medianKernel<<<grid, block>>>(d_in, d_out, width, height, step);
        cudaDeviceSynchronize();


    cudaMemcpy(out.data, d_out, height*step, cudaMemcpyDeviceToHost);
    
    cv::imwrite("output.png", out);
    cv::imshow("Output",out);

    cv::waitKey(0);
    cv::destroyAllWindows();
    cudaFree(d_in);
    cudaFree(d_out);
}


/*

Questo è indifferente perché anche se si fa 100 volte alla fine quello che fa il kernel
è prendere una finestra di 3x3 (quindi 9 elementi) per ogni thread
e ordinarli scambiandoli . Quindi sono sempre quelli e sono indipendenti ai cicli.
Forse può cambiare qualcosa sicuramente dimensionando la finestra

    //
    int v=100;
    while(v>=0)
    {
        printf("index : %d \n",v);
    
        medianKernel<<<grid, block>>>(d_in, d_out, width, height, step);
        cudaDeviceSynchronize();
    
        cudaMemcpy(d_in,d_out,height*step,cudaMemcpyDeviceToHost);
    
        medianKernel<<<grid, block>>>(d_in, d_out, width, height, step);
        cudaDeviceSynchronize();
        cudaMemcpy(d_in,d_out,height*step,cudaMemcpyDeviceToHost);
    
        medianKernel<<<grid, block>>>(d_in, d_out, width, height, step);
        cudaDeviceSynchronize();
        cudaMemcpy(d_in,d_out,height*step,cudaMemcpyDeviceToHost);
        v--;    
    }
    
    
    
    
    
    
    //


*/
