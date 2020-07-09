
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define SIZE (100*1024*1024)


float cuda_malloc_test(int size, bool up, bool hostAlloc = false) {
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (hostAlloc) {
		a = (int*)cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault);
	} else {
		a = (int*)malloc(size * sizeof(*a));
	}
	cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; i++) {
		if (up)
			cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		else
			cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	if (hostAlloc) {
		cudaFreeHost(dev_a);
	} else {
		cudaFree(dev_a);
	}
	cudaFree(a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsedTime;		
}

int main()
{
	float elapsedTime;
	
	printf("cudaMalloc test:\n");
	float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
	elapsedTime = cuda_malloc_test(SIZE, true);
	printf("Time using cudaMalloc(up): %3.1f ms.\n", elapsedTime);
	printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
	elapsedTime = cuda_malloc_test(SIZE, false);
	printf("Time using cudaMalloc(down): %3.1f ms.\n", elapsedTime);
	printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

	printf("cudaHostalloc test:\n");
	elapsedTime = cuda_malloc_test(SIZE, true, 1);
	printf("Time using cudaMalloc(up): %3.1f ms.\n", elapsedTime);
	printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
	elapsedTime = cuda_malloc_test(SIZE, false, 1);
	printf("Time using cudaMalloc(down): %3.1f ms.\n", elapsedTime);
	printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
}
