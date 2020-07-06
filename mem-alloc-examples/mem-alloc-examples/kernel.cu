
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	int * dev_a  = 0;

	printf("1. cudaMalloc example, default parameters.\n");
	printf("dev_a before: 0x%08x\n", dev_a);
	cudaMalloc((void**)&dev_a, 1024);
	printf("dev_a after: 0x%08x\n", dev_a);
	cudaFree(dev_a);
	getchar();
	return 0;
}
