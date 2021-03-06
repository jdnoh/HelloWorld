#include <iostream>
__global__ void add_constant(int n, int *x)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    x[tid] += n;
}

int main(void)
{
    int dNum = 1<<24;
    int *x, *d_x, nT=32;
    size_t memSize = sizeof(int)*dNum;

    // memory allocation in the host and initialization
    x = (int *)malloc(memSize);
    for(int i=0; i<dNum; i++) x[i] = i;

    // memory allocation in the device
    cudaMalloc(&d_x, memSize);
    // copy from host to device
    cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);

    // kernel launch
    add_constant<<<dNum/nT, nT>>>(1, d_x);

    printf("%d\n",x[0]);
    // copy from device to host
    cudaMemcpy(x, d_x, memSize, cudaMemcpyDeviceToHost);
    printf("%d\n",x[0]);

    return 0;
} 
