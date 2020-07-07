
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel(int * dev_a, int * pMemAddr)
{
    *dev_a = 1000;
    *pMemAddr = 1001;
}

int main()
{
	int * dev_a;
	int * a;
	int memAddr = 0;
	int stat;

	stat = cudaMalloc(&dev_a, sizeof(int));
	a = (int*)malloc(sizeof(int));

	if (a != NULL) {
		*a = 200;
	} else {
		printf("Failure allocating for a...\n");
		return 1;
	}
	printf("Size of int: %d.\n", sizeof(int));
	
	printf("a before kernel call: %u.\n", *a);
	printf("1. cudaMalloc example, default parameters.\n");
	printf("dev_a host address before: 0x%08x\n", dev_a);
	cudaMalloc((void**)&dev_a, sizeof(int));
	printf("dev_a host address after: 0x%08x\n", dev_a);

	cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice);
	kernel <<<1, 1>>>  (dev_a, &memAddr);
	cudaMemcpy(&a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

	printf("a after kernel call: %u.\n", *a);
	//printf("memAddr %08x", memAddr);

	cudaFree(dev_a);
	getchar();
	return 0;
}
