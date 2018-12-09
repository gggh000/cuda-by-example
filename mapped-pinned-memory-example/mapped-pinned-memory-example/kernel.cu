
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define N 10

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;								// current thread's x dimension.

	if (tid < N)
		c[tid] = a[tid] + b[tid];						// add as long as it is smaller than input vector.,	
}

int main()
{
	//int a[N], b[N], c[N];
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;
	int stat[2];

	dev_a = NULL;
	dev_b = NULL;
	dev_c = NULL;

	// Start allocating memory for 3 vectors in GPU.

	stat[0] = cudaHostAlloc((void**)&a, N * sizeof(int), cudaHostAllocMapped);
	stat[1] = cudaHostAlloc((void**)&b, N * sizeof(int), cudaHostAllocMapped);
	stat[2] = cudaHostAlloc((void**)&c, N * sizeof(int), cudaHostAllocMapped);

	printf("\n1. stat: %d", stat[0]);
	printf("\n1. stat: %d", stat[1]);
	printf("\n1. stat: %d", stat[2]);

	//checkCudaErrors(cudaHostGetDevicePointer((void **)&dev_a, (void*)a, 0));
	//checkCudaErrors(cudaHostGetDevicePointer((void **)&dev_b, (void*)b, 0));
	//checkCudaErrors(cudaHostGetDevicePointer((void **)&dev_c, (void*)c, 0));
	
	printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);
	stat[0] = cudaHostGetDevicePointer((void **)&dev_a, (void*)a, 0);
	stat[1] = cudaHostGetDevicePointer((void **)&dev_b, (void*)b, 0);
	stat[2] = cudaHostGetDevicePointer((void **)&dev_c, (void*)c, 0);
	printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);

	printf("\n2. stat: %d", stat[0]);
	printf("\n2. stat: %d", stat[1]);
	printf("\n2. stat: %d", stat[2]);

	printf("\n0x%08x", a);
	printf("\n0x%08x", dev_a);

	// Construct vectors values for a and b vector.
	
	for (int i = 0; i < N; i++) {
		dev_a[i] = i;
		dev_b[i] = i*i;
	}
	
	// Copy the summing vectors to device. 

	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <1, N >> > (dev_a, dev_b, dev_c);

	// Copy the summed vector back to host.

	//cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print the vector now.

	for (int i = 0; i < N; i++)
		printf("\n%d + %d = %d", dev_a[i], dev_b[i], dev_c[i]);

	// Release device memory. 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	getchar();
	return 0;
}
