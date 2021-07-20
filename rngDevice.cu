#include <random>
#include <cuda.h>
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

__global__ void initPRNG(int seed, curandState *rngState)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, tid, 0, &rngState[tid]);
}

__global__ void generate_uniform(float *data, curandState *rngState)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    curandState localState = rngState[tid];
    data[tid] = curand_uniform(&localState);
    rngState[tid] = localState;
}

int main()
{
    int n=1<<24;
    int nThreads = 128, nBlocks = n / nThreads;

    std::random_device rd;
    unsigned int seed = rd();
    // unsigned int seed = time(0);
    printf("seed = %u\n", seed);
    
    // default PRNG 
    curandState *rngState_dev;
    cudaMalloc(&rngState_dev, sizeof(curandState)*nBlocks*nThreads);
    initPRNG<<<nBlocks, nThreads>>>(seed, rngState_dev);

    float *hostData, *devData;
    size_t memSize=sizeof(float)*n;
    // host and device memory allocation
    hostData = (float *)malloc(memSize);
    cudaMalloc(&devData, memSize);

    generate_uniform<<<nBlocks, nThreads>>>(devData, rngState_dev);
    
    cudaMemcpy(hostData, devData, memSize, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++) printf("%d %e\n", i, hostData[i]);
    
    cudaFree(rngState_dev); 
    cudaFree(devData); free(hostData);
}

