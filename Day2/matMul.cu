#include <iostream>
#include <cuda_runtime.h>
#include<stdlib.h>

#define M 15
#define N 17
#define K 20


__global__ void matrix_multiplication_kernel(const int* A, const int* B, int* C, int p, int q, int r) {

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ( (row<p) && (col<r) ){
        C[row*r+col] = 0;
        for (int n = 0;n<q;n++)
          C[row*r+col]  += A[row*q+n] * B[n*r+col];
    }
    
}

void matMul(int* A, int* B, int* C, int p, int q, int r){

    // Device variable
    int *A_d, *B_d, *C_d;
    int memSizeOfA = p*q*sizeof(int);
    int memSizeOfB = q*r*sizeof(int);
    int memSizeOfC = p*r*sizeof(int);

    // Memory Allocation on Device
    cudaMalloc((void**)&A_d,memSizeOfA);
    cudaMalloc((void**)&B_d,memSizeOfB);
    cudaMalloc((void**)&C_d,memSizeOfC);

    // Input Data Transfer to Device
    cudaMemcpy(A_d,A, memSizeOfA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B, memSizeOfB, cudaMemcpyHostToDevice);
    printf("Input Data Transfer Successfull !!\n");

    // dimension of grid and block
    dim3 dimGrid(4,3);
    dim3 dimBlock(6,6);

    // Kernel Call
    matrix_multiplication_kernel<<< dimGrid, dimBlock>>>(A_d, B_d, C_d, p, q, r);
    printf("Kernel Vector Addition Successfull !!\n");

    // Output Data Transfer to Host
    cudaMemcpy(C, C_d, memSizeOfC, cudaMemcpyDeviceToHost);
    printf("Output Data Transfer Successfull !!\n");

    // Free Cuda Memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);


}

int main() {
   
    int A[M][N];
    int B[N][K];
    int C[M][K];

    // initialization
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            A[i][j] = (rand()%50) + 5; 
    
    for(int i=0;i<N;i++)
        for(int j=0;j<K;j++)
            B[i][j] = (rand()%5) + 10;


    // Matrix Multiplication
    matMul((int *)A, (int *)B, (int *)C, M, N, K);

    // Printing Matrices
    for(int i=0;i<10;i++)
        printf("-");
    printf("Matrix A");
    for(int i=0;i<10;i++)
        printf("-");
    printf("\n");

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++)
            printf("%d ",A[i][j]);
        printf("\n");
    }
    printf("\n\n\n"); 
    
    for(int i=0;i<10;i++)
        printf("-");
    printf("Matrix B");
    for(int i=0;i<10;i++)
        printf("-");
    printf("\n");

    for(int i=0;i<N;i++){
        for(int j=0;j<K;j++)
            printf("%d ",B[i][j]);
        printf("\n");
    }
    printf("\n\n\n"); 

    for(int i=0;i<10;i++)
        printf("-");
    printf("Matrix C");
    for(int i=0;i<10;i++)
        printf("-");
    printf("\n");

    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++)
            printf("%d ",C[i][j]);
        printf("\n");
    }
}

