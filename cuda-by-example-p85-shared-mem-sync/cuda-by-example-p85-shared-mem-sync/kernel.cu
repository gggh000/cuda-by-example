#include <book.h>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {

	__shared__ float cache(threadsPerBlock);
	int tid = threadIdx.x + blockIdx.x + blockDim.x;				// index of thread within current grid.
	int cacheIndex = threadIdx.x;									// index of thread within current block.

	float temp = 0;

	while (tid < N) {
		temp += a[tid] * b[tid];									// dot product to temp.
		tid += blockDim.x + gridDim.x;								// increase by grid size.
	}

	// Set the cache values

	cache[cacheIndex] = temp;										// store it in the current threadidx in block.

	// Synchronize threads in this block.

	__syncthreads();

	// For reductions, threadsPerBlock must be a power of 2 because of 
	// the following code.

	int i = blockDim.x / 2;

	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];

		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main()
{
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	// Allocate memory on the CPU side.

	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float *)malloc(blockPerGrid * sizeof(float));

	// allocate memory on the GPU.

	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

	// File in the host memory with data.

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	// Copy the arrays a and b to the GPU

	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	cudaMemcpy(partial_c, dev_partial_c, blockPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	// Finish up on the CPU side.

	c = 0;

	for (int i = 0; i < blockPerGrid; i++) {
		c += partial_c[i];
	}

	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

	printf("Does GPU value $.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

	// free memory on the GPU side.

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	// free memory on the CPU side.

	free(a);
	free(b);
	free(partial_c);

	getchar();
	return 0;

}