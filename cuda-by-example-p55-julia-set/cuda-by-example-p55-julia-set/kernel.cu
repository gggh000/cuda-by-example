
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <c:\book\CUDA-By-Example\common\cpu_bitmap.h>
#include <c:\book\CUDA-By-Example\common\book.h>

#include <stdio.h>

#define DIM 1000

__device__ struct cuComplex {
	float r;
	float i;

	// copy constructor.s

	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}

	__device__ cuComplex operator * (const cuComplex & a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator + (const cuComplex & a) {
		return cuComplex(r + a.r, i + a.i);
	}
};


__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;

	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if(a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel(unsigned char * ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// Now calculate the value at the position.

	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char * dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	dim3 grid(DIM, DIM);
	kernel<<<grid, 1 >>> (dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}
