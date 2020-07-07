
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kerneladd(int  *dev_c)
{
    *dev_c = 1000;
//    *pMemAddr = 1001;
}

int main()
{
	int * dev_c;
	int a = 300;
	int memAddr = 0;
	int stat;
	
	printf("Size of int: %d.\n", sizeof(int));
	printf("a before kernel call: %u.\n", a);
	printf("1. cudaMalloc example, default parameters.\n");
	//printf("dev_c host address before cudaMalloc: 0x%08x\n", dev_c);

	cudaMalloc((void**)&dev_c, sizeof(int));

	//printf("dev_c host address after cudaMalloc: 0x%08x\n", dev_c);

	cudaMemcpy(dev_c, &a, sizeof(int), cudaMemcpyHostToDevice);
	kerneladd <<<1, 1>>>  (dev_c);
	cudaMemcpy(&a, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("a after kernel call: %u.\n", a);
	//printf("memAddr %08x", memAddr);

	cudaFree(dev_c);
	getchar();
	return 0;
}
