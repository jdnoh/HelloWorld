#include <cuda.h>
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

__global__ void initPRNG(int seed, curandState *rngState)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, tid, 0, &rngState[tid]);
}

__global__ void generate_uniform_int(int n, unsigned int *data, int q, curandState *rngState)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned nGrid = blockDim.x*gridDim.x;
    curandState localState = rngState[tid];

    for(int i=tid; i<n; i+= nGrid) 
        data[i] = curand(&localState)%q;

    rngState[tid] = localState;
}

int main()
{

    int nBlocks = 128, nThreads = 128;
    int seed = 1234;
    int q=4;
    int hist[q];
    
    // default PRNG 
    curandState *rngState_dev;
    cudaMalloc(&rngState_dev, sizeof(curandState)*nBlocks*nThreads);
    initPRNG<<<nBlocks, nThreads>>>(seed, rngState_dev);

    int   n=1<<28;
    unsigned int *hostData, *devData;
    size_t memSize=sizeof(unsigned int)*n;
    // host and device memory allocation
    hostData = (unsigned int *)malloc(memSize);
    cudaMalloc(&devData, memSize);

    for(int i=0; i<10; i++) 
        generate_uniform_int<<<nBlocks, nThreads>>>(n, devData, q, rngState_dev);
    
    cudaMemcpy(hostData, devData, memSize, cudaMemcpyDeviceToHost);

    // histogram
    for(int i=0; i<q; i++) hist[i] = 0;
    for(int i=0; i<n; i++) {
        hist[hostData[i]] ++;
    }
    for(int i=0; i<q; i++) printf("%d %d\n", i, hist[i]);

    thrust::sort(thrust::device_ptr<unsigned int>(devData),
            thrust::device_ptr<unsigned int>(devData)+n);

    thrust::device_vector<unsigned int> histogram(q,0);
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::upper_bound(thrust::device_ptr<unsigned int>(devData),
            thrust::device_ptr<unsigned int>(devData)+n,
            search_begin, search_begin+q, histogram.begin());
    thrust::adjacent_difference(histogram.begin(), histogram.end(),
            histogram.begin());

    for(int i=0; i<histogram.size(); i++)
        std::cout << i << " " << histogram[i] << "\n";
    
    cudaFree(rngState_dev); 
    cudaFree(devData); free(hostData);
}

