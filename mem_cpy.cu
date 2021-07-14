#include <iostream>
__global__ void add_constant(int n, int *x)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    x[tid] *= 2;
}

int main(void)
{
    int dNum = 1<<24;
    int *x, *d_x, nT=32;
    size_t memSize = sizeof(int)*dNum;

    x = (int *)malloc(memSize);
    for(int i=0; i<dNum; i++) x[i] = i;

    cudaMalloc(&d_x, memSize);
    cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);
    add_constant<<<dNum/nT, nT>>>(dNum, d_x);
    cudaMemcpy(x, d_x, memSize, cudaMemcpyDeviceToHost);

    for(int i=0; i<dNum; i++) if(x[i]!=2*i) printf("Error\n");
    return 0;
} 
