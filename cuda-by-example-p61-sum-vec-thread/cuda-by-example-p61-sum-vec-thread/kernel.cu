
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define N 1000

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;								// current thread's x dimension.

	if (tid < N)
		c[tid] = a[tid] + b[tid];						// add as long as it is smaller than input vector.,	
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	int stat;

	// Start allocating memory for 3 vectors in GPU.

	stat = cudaMalloc((void**)&dev_a, N * sizeof(int));
	stat = cudaMalloc((void**)&dev_b, N * sizeof(int));
	stat = cudaMalloc((void**)&dev_c, N * sizeof(int));

	// Construct vectors values for a and b vector.

	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
	}

	// Copy the summing vectors to device. 

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <1, N>> > (dev_a, dev_b, dev_c);

	// Copy the summed vector back to host.

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print the vector now.

	for (int i = 0; i < N; i++)
		printf("\n%d + %d = %d", a[i], b[i], c[i]);

	// Release device memory. 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	getchar();
	return 0;
}
