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
    int seed = time(0);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    float *hostData, *devData;
    int   n=1<<24;
    size_t memSize=sizeof(float)*n;
    // host and device memory allocation
    hostData = (float *)malloc(memSize);
    cudaMalloc(&devData, memSize);

    // generate n random numbers in (0,1] on the device array 
    for(int i=0; i<10; i++) 
        curandGenerateUniform(gen, devData, n);

    cudaMemcpy(hostData, devData, memSize, cudaMemcpyDeviceToHost);

    // find the minimum/maximum random number
    float rmax = 0.0, rmin = 1.0;
    for(int i=0; i<n; i++) {
        if(hostData[i]>rmax) rmax = hostData[i] ;
        if(hostData[i]<rmin) rmin = hostData[i] ;
    }
    
    printf("miminum = %e, maximum = %e\n", rmin, rmax) ;
    curandDestroyGenerator(gen);
    cudaFree(devData); free(hostData);
}

