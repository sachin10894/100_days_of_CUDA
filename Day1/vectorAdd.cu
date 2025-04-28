#include <iostream>
#include <cuda_runtime.h>
#include<stdlib.h>
#include <math.h>

#define BLOCKSIZE 256.0
#define VECTORSIZE 1000

__global__ 
void vecAddKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
        c[i] = a[i] + b[i];
}

void vectorAdd(float *a_h, float *b_h, float *c_h, int n){
    
    // Device variable
    float *a_d, *b_d, *c_d;
    int memSizeOfVec = n*sizeof(float);

    // Memory Allocation on Device
    cudaMalloc((void**)&a_d,memSizeOfVec);
    cudaMalloc((void**)&b_d,memSizeOfVec);
    cudaMalloc((void**)&c_d,memSizeOfVec);

    // Input Data Transfer to Device
    cudaMemcpy(a_d,a_h, memSizeOfVec, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h, memSizeOfVec, cudaMemcpyHostToDevice);
    printf("Input Data Transfer Successfull !!\n");

    // Kernel Call
    vecAddKernel<<< ceil(VECTORSIZE/BLOCKSIZE), (int)BLOCKSIZE>>>(a_d, b_d, c_d, n);
    printf("Kernel Vector Addition Successfull !!\n");

    // Output Data Transfer to Host
    cudaMemcpy(c_h, c_d, memSizeOfVec, cudaMemcpyDeviceToHost);
    printf("Output Data Transfer Successfull !!\n");

    // Free Cuda Memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    
    float a[VECTORSIZE], b[VECTORSIZE], c[VECTORSIZE];
    int n = VECTORSIZE;

    for(int i =0; i< n; i++){
        a[i] = (rand()%10) + 1;
        b[i] = (rand()%10) + 1;
    }
    printf("Initialised Successfully !!\n");

    vectorAdd(a,b,c,n);
    printf("Addition Performed Successfully !!\n");

    printf("vec a   +   vec b  =   vec c\n");
    for (int i =0; i< n; i++)
        printf("%f\t+\t%f\t=\t%f\n",a[i],b[i],c[i]);
       
    return 0;
}
