#include <iostream>

// Kernel function to add the elements of two arrays
__global__ void add(int n, int *x, int *y, int a)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<n) x[tid] = a*x[tid] + y[tid];
}

int main(void)
{
    int dNum = 1<<24;
    int *x, *y;
    // memory size for each array
    size_t memSize = sizeof(int)*dNum;

    // unified memory allocation
    cudaMallocManaged(&x, memSize); 
    cudaMallocManaged(&y, memSize);

    // initialization in CPU
    for(int i=0; i<dNum; i++) { x[i] = 1; y[i] = 2; }

    // device id
    int gpuId;
    cudaGetDevice(&gpuId);

    // pre-fetch 'x' and 'y' to the device
    cudaMemPrefetchAsync(x, memSize, gpuId);
    cudaMemPrefetchAsync(y, memSize, gpuId);

    // grid configuration (dNum = numBlocks * numThreads)
    int numThreads = 512;
    int numBlocks  = (dNum + numThreads-1) / numThreads;

    // Run kernel on the device
    add<<<numBlocks, numThreads>>>(dNum, x, y, 2);
    // explicit barrier
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(x, memSize, cudaCpuDeviceId);
    cudaMemPrefetchAsync(y, memSize, cudaCpuDeviceId);
    for(int i=0; i<dNum; i++) if(x[i]!=4) printf("Error!\n");

    // Free memory
    cudaFree(x); cudaFree(y);
    return 0;
}
