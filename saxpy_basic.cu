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
    int *x, *y, *d_x, *d_y;
    // memory size for each array
    size_t memSize = sizeof(int)*dNum;

    // memory allocation in the host (CPU) and initialization
    x = (int *)malloc(memSize); 
    y = (int *)malloc(memSize);
    for (int i = 0; i < dNum; i++) { x[i] = 1; y[i] = 2; }

    // memory allocation in the device (GPU)
    cudaMalloc(&d_x, memSize); 
    cudaMalloc(&d_y, memSize);

    // copy from the host to the device (CPU -> GPU)
    cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, memSize, cudaMemcpyHostToDevice);

    // grid configuration (dNum = numBlocks * numThreads)
    int numThreads = 512;
    int numBlocks  = (dNum + numThreads-1) / numThreads;

    // Run kernel on the device
    add<<<numBlocks, numThreads>>>(dNum, d_x, d_y, 2);

    // copy back from the devide to the host (GPU -> CPU)
    cudaMemcpy(x, d_x, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, memSize, cudaMemcpyDeviceToHost);

    // validating
    for(int i=0; i<dNum; i++) if(x[i] != 4) printf("Error!\n");

    // Free memory
    free(x); free(y); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
