
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	cudaDeviceProp prop;
	int dev;
	int stat; 

	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);

	// causes build error. 
	memset ( &prop, 0, sizeof(cudaDeviceProp));

	prop.major = 1;
	prop.minor = 3;
	cudaChooseDevice(&dev, &prop);
	printf("ID of current CUDA device closest to revision: %d\n", dev);

	printf("\nComputeMode: %d", prop.computeMode);
	printf("\ncanMapHostMemory: %d", prop.canMapHostMemory);
	printf("\nBus Device Domain: %d-%d-%d", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);

	getchar();
 return 0;
}
