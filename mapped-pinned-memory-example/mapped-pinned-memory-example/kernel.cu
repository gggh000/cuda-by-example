
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define N 10
#define CONFIG_ENABLE_PINNED_MAPPED_MEMO 0

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;								// current thread's x dimension.
	printf("\nadd(): tid: %d. 0x%08x, 0x%08x, 0x%08x %d, %d, %d.", tid, a, b, c, *a, *b, *c);
	if (tid < N) {
		a[tid] += 100;
		c[tid] = a[tid] + b[tid];						// add as long as it is smaller than input vector.,	
		//c[tid] = (int)&c[tid];
	}
}

int main()
{
#if CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1
	int *a, *b, *c;
#else
	int a[N], b[N], c[N];
#endif

	int *dev_a, *dev_b, *dev_c;
	int stat[2];

	dev_a = NULL;
	dev_b = NULL;
	dev_c = NULL;

	// Start allocating memory for 3 vectors in GPU.

	if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1) {
		printf("CONFIG_ENABLE_PINNED_MAPPED_MEMO on...");
		stat[0] = cudaHostAlloc((void**)&a, N * sizeof(int), cudaHostAllocDefault);
		stat[1] = cudaHostAlloc((void**)&b, N * sizeof(int), cudaHostAllocDefault);
		stat[2] = cudaHostAlloc((void**)&c, N * sizeof(int), cudaHostAllocDefault);
	}	else {
		printf("CONFIG_ENABLE_PINNED_MAPPED_MEMO off...");
		stat[0] = cudaMalloc((void**)&dev_a, N * sizeof(int));
		stat[1] = cudaMalloc((void**)&dev_b, N * sizeof(int));
		stat[2] = cudaMalloc((void**)&dev_c, N * sizeof(int));
	}

	for (int i = 0; i < 3; i++)
		printf("\n%d. stat: %d", i, stat[i]);

	if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1) {
		printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);
		stat[0] = cudaHostGetDevicePointer((void **)&dev_a, (void*)a, 0);
		stat[1] = cudaHostGetDevicePointer((void **)&dev_b, (void*)b, 0);
		stat[2] = cudaHostGetDevicePointer((void **)&dev_c, (void*)c, 0);
		printf("\ndev_a:c: 0x%08x, 0x%08x, 0x%08x", dev_a, dev_b, dev_c);

		for (int i = 0; i < 3; i++)
			printf("\n%d. stat: %d", i, stat[i]);
		printf("\n0x%08x", a);
		printf("\n0x%08x", dev_a);
	}

	// Construct vectors values for a and b vector.
	
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i*i;
		c[i] = -1;
	}
	
	// Copy the summing vectors to device. 

	if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 0) {
		cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	}
	printf("\nResult before:");
	for (int i = 0; i < N; i++) {
		if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1) {
			printf("\n%d + %d = %d", dev_a[i], dev_b[i], dev_c[i]);
		}
		else {
			printf("\n%d + %d = %d", a[i], b[i], c[i]);
		}
	}

	add <<<1, N >>> (dev_a, dev_b, dev_c);

	// Copy the summed vector back to host.

	// Print the vector now.

	if (CONFIG_ENABLE_PINNED_MAPPED_MEMO != 1) {
		cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	printf("\nResult before:");
	for (int i = 0; i < N; i++) {
		if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1) {
			printf("\n%d + %d = %d", dev_a[i], dev_b[i], dev_c[i]);
		} else {
			printf("\n%d + %d = %d", a[i], b[i], c[i]);
		}
	}

	// Release device memory. 

	if (CONFIG_ENABLE_PINNED_MAPPED_MEMO == 1) {
		cudaFreeHost(dev_a);
		cudaFreeHost(dev_b);
		cudaFreeHost(dev_c);
	} else {
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	}

	getchar();
	return 0;
}
