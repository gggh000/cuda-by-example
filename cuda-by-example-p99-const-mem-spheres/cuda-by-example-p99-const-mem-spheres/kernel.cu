
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <c:\book\CUDA-By-Example\common\cpu_bitmap.h>
#include <c:\book\CUDA-By-Example\common\book.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#define ENABLE_CONST_MEM_P104 0

/*
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include "book.h"
*/
#define DIM 1024
#define rnd(x) (x*rand()/ RAND_MAX)
#define SPHERES 20
#define INF 2e10f

struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;

	// hit functions. If ox, oy given coordinate has distance from the x,y center of sphere which is 
	// smaller than its radius, we get hit, in that case, calclulate distance sqrt((dz) from r^2 - (dx^2+dy^2))
	// and return dz + z, that z-coordinate of spheres center plus distance. 

	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

//Sphere * s;
//__constant__ Sphere s[SPHERES];

__global__ void kernel(Sphere *s, unsigned char * ptr) {
	// map from threadIdx/blockIdx to pixel positions.

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = 0, g = 0, b = 0;

	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++) {
		float n, t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;

}


int main()
{
	// capture the start time.

	/*CudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(&start, 0);
	*/
	Sphere * s;

	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	// alloc memory on the GPU for output bitmap.

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	// alloc memory for sphere dataset.

	cudaMalloc((void**)&s, (size_t)sizeof(Sphere) * SPHERES);

	// alloc temp memory, initialize it, copy to memory on the GPU,
	// and then free our temp memory.

	Sphere * temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
	free(temp_s);

	// generate bitmap from our sphere sets

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (s, dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
	cudaFree(s);
}

