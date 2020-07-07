
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kerneladd(int  *dev_c, int * pMemAddr)
{
    *dev_c = 1000;
    *pMemAddr = (int)dev_c;
}

int main()
{
	int * dev_c;
	int a = 300;
	int memAddr = 0;
	int * dev_memAddr;
	int stat;
	
	printf("Disjoint global memory example. In such a situation, each GPU and CPU has its own separate address space. \
This is examplified here as memAddr to hold the address of the dev_c from CPU (host) side and dev_memAddr is to hold the GPU (device) \
side of the dev_c.\n");
	printf("Size of int: %d.\n", sizeof(int));
	printf("a before kernel call: %u.\n", a);
	printf("1. cudaMalloc example, default parameters.\n");
	
	cudaMalloc((void**)&dev_c, sizeof(int));
	cudaMalloc((void**)&dev_memAddr, sizeof(int));
	printf("dev_c (host side): 0x%08x address after cudaMalloc.\n", dev_c);
	cudaMemcpy(dev_c, &a, sizeof(int), cudaMemcpyHostToDevice);
	kerneladd <<<1, 1>>>  (dev_c, dev_memAddr);
	cudaMemcpy(&a, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&memAddr, dev_memAddr, sizeof(int), cudaMemcpyDeviceToHost);

	printf("a after kernel call: %u.\n", a);
	printf("dev_c (GPU side): 0x%08x after kernel call.\n", memAddr);

	cudaFree(dev_c);
	getchar();
	return 0;
}
