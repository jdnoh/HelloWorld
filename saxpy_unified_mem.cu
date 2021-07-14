#include <iostream>

// Kernel function to add the elements of two arrays
__global__ void add(int n, int *x, int *y, int a)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<n) x[tid] = a*x[tid] + y[tid];
}

int main(void)
{
    int dNum = 1<<20;
    int *x, *y;
    // memory size for each array
    size_t memSize = sizeof(int)*dNum;

    // unified memory allocation
    cudaMallocManaged(&x, memSize); 
    cudaMallocManaged(&y, memSize);

    // initialization in CPU
    for(int i=0; i<dNum; i++) { x[i] = 1; y[i] = 2; }

    // grid configuration (dNum = numBlocks * numThreads)
    int numThreads = 512;
    int numBlocks  = (dNum + numThreads-1) / numThreads;

    // Run kernel on the device
    add<<<numBlocks, numThreads>>>(dNum, x, y, 2);
    // explicit barrier
    cudaDeviceSynchronize();

    for(int i=0; i<dNum; i++) if(x[i]!=4) printf("Error!\n");

    // Free memory
    cudaFree(x); cudaFree(y);
    return 0;
}
