#include <random>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>
#include <time.h>

int main()
{
    
    curandGenerator_t gen;
    // default (WOWXOR) or Mersenne-Trister pseudo random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937); 

    // initialize the PRNG with seed
    // std::random_device rd;
    // unsigned int seed = rd();
    unsigned int seed = time(0);
    printf("seed = %u\n", seed);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    float *hostData, *devData;
    int   n=1<<24;
    size_t memSize=sizeof(float)*n;
    // host and device memory allocation
    hostData = (float *)malloc(memSize);
    cudaMalloc(&devData, memSize);

    // generate n random numbers in (0,1] on the device array 
    curandGenerateUniform(gen, devData, n);

    cudaMemcpy(hostData, devData, memSize, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++) printf("%d %e\n", i, hostData[i]);

    curandDestroyGenerator(gen);
    cudaFree(devData); free(hostData);
}

