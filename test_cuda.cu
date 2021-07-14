#include <cuda.h>
#include <stdio.h>
#include <curand.h>
int main() {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 
	curandSetPseudoRandomGeneratorSeed(gen, 1234);
	float *hostData, *devData;
	hostData = (float *)malloc(2*sizeof(float));
	cudaMalloc(&devData, 2*sizeof(float));
	curandGenerateUniform(gen, devData, 2);
	cudaMemcpy(hostData, devData, 2*sizeof(float), cudaMemcpyDeviceToHost);
	printf("ran1 = %e\nran2 = %e\n", hostData[0], hostData[1]);
	curandDestroyGenerator(gen); cudaFree(devData); free(hostData);
}
