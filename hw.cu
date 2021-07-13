// #include "iostream"
#include "stdio.h"

// Kernel function to add the elements of two arrays
__global__ void hello_fromGPU(int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Hello World from thread %d-%d\n",n,tid);
    return ;
}

void hello_fromCPU()
{
    printf("Hello World from CPU\n");
    return ;
}

int main(void)
{
    hello_fromGPU<<<4, 3>>>(0);
    hello_fromGPU<<<4, 3>>>(1);
    // cudaDeviceSynchronize();
    hello_fromCPU();
    return 0;
}
