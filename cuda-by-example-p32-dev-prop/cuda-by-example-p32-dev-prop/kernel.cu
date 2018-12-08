
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main()
{
	cudaDeviceProp prop;
	int dev;
	int stat;
	int count; 
	int i;

	cudaGetDeviceCount(&count);								// count is updated with No. of GPU-s.

	for (i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("\n--- General information for device %d ---", i);
		printf("\nName: %s.", prop.name);
		printf("\nCompute capability: %d.%d", prop.major, prop.minor);
		printf("\nCompute mode: 0x%x", prop.computeMode);
		printf("\nClock rate: %d", prop.clockRate);
		printf("\nDevice copy overlap: ", prop.deviceOverlap);
		printf("\nKernel execution timeout: %d", prop.kernelExecTimeoutEnabled);
		printf("\naSync engine count: %d", prop.asyncEngineCount);
		printf("\nConcurrent kernels: %d", prop.concurrentKernels);
		printf("\nCan map host memory: %d", prop.canMapHostMemory);
		printf("\nPCI Bus Device Domain: %d %d %d", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
		printf("\nTotal global memory: 0x%x", prop.totalGlobalMem);
		printf("\nTotal const memory: 0x%x", prop.totalConstMem);
		printf("\nTotal shared memory/block: 0x%x", prop.sharedMemPerBlock);
		printf("\nTotal shared memory/multiprocessor: 0x%x", prop.sharedMemPerMultiprocessor);
		printf("\nMemory bus width: %d", prop.memoryBusWidth);
		
		printf("\nintegated: ", prop.integrated);
		printf("\nmaxGridSize: 0%d", prop.maxGridSize);
		printf("\nmaxThreadsDim: 0x%x", prop.maxThreadsDim);
		printf("\nmaxThreadsPerBlock: %d ", prop.maxThreadsPerBlock);
		printf("\nmaxThreadsPerMultiProcessor: %d", prop.maxThreadsPerMultiProcessor);
		printf("\nmultiProcessorCount: %d", prop.multiProcessorCount);







	}


	getchar();
	return 0;
}
