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

__global__ void generate_uniform(int n, float *data, curandState *rngState)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned nGrid = blockDim.x*gridDim.x;
    curandState localState = rngState[tid];

    for(int i=tid; i<n; i+= nGrid) 
        data[i] = curand_uniform(&localState);

    rngState[tid] = localState;
}

int main()
{

    int nBlocks = 128, nThreads = 128;
    int seed = time(0);
    
    // default PRNG 
    curandState *rngState_dev;
    cudaMalloc(&rngState_dev, sizeof(curandState)*nBlocks*nThreads);
    initPRNG<<<nBlocks, nThreads>>>(seed, rngState_dev);

    int   n=1<<24;
    float *hostData, *devData;
    size_t memSize=sizeof(float)*n;
    // host and device memory allocation
    hostData = (float *)malloc(memSize);
    cudaMalloc(&devData, memSize);

    for(int i=0; i<10; i++) 
        generate_uniform<<<nBlocks, nThreads>>>(n, devData, rngState_dev);
    
    cudaMemcpy(hostData, devData, memSize, cudaMemcpyDeviceToHost);

    // find the minimum/maximum random number
    float rmax = 0.0, rmin = 1.0;
    for(int i=0; i<n; i++) {
        if(hostData[i]>rmax) rmax = hostData[i] ;
        if(hostData[i]<rmin) rmin = hostData[i] ;
    }
    
    printf("miminum = %e, maximum = %e\n", rmin, rmax) ;
    cudaFree(rngState_dev); 
    cudaFree(devData); free(hostData);
}

