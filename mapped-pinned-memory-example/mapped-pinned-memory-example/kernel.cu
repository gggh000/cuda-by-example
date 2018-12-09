
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define N 10

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;								// current thread's x dimension.

	if (tid < N) {
		c[tid] = a[tid] + b[tid];						// add as long as it is smaller than input vector.,	
		//c[tid] = (int)&c[tid];
	}
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

	for (int i = 0; i < 3; i++)
		printf("\n%d. stat: %d", i, stat[i]);

	printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);
	stat[0] = cudaHostGetDevicePointer((void **)&dev_a, (void*)a, 0);
	stat[1] = cudaHostGetDevicePointer((void **)&dev_b, (void*)b, 0);
	stat[2] = cudaHostGetDevicePointer((void **)&dev_c, (void*)c, 0);
	printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);

	for (int i = 0; i < 3; i ++)
		printf("\n%d. stat: %d", i, stat[i]);

	printf("\n0x%08x", a);
	printf("\n0x%08x", dev_a);

	// Construct vectors values for a and b vector.
	
	for (int i = 0; i < N; i++) {
		dev_a[i] = i;
		dev_b[i] = i*i;
		dev_c[i] = -1;
	}
	
	// Copy the summing vectors to device. 

	//cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <1, N >> > (dev_a, dev_b, dev_c);

	// Copy the summed vector back to host.

	//cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print the vector now.

	for (int i = 0; i < N; i++) {
		printf("\n%d + %d = %d", dev_a[i], dev_b[i], dev_c[i]);
		//printf("\n%d + %d = %d", a[i], b[i], c[i]);
	}
	// Release device memory. 
	cudaFreeHost(dev_a);
	cudaFreeHost(dev_b);
	cudaFreeHost(dev_c);

	getchar();
	return 0;
}
