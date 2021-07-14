#include "iostream"
__global__ void hello_fromGPU(int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Hello World from thread %d-%d\n", n, tid);
}
void hello_fromCPU()
{
    printf("Hello World from CPU\n");
}

int main()
{
    hello_fromGPU<<<2,3>>>(0);
    // hello_fromGPU<<<2,3>>>(1);
    hello_fromCPU();
    return 0;
}
