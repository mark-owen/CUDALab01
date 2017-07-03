// -*-c-*-
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);



__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < max) c[i] = a[i] + b[i];

}

void vectorAddCPU(int n, int* a, int*b, int*c) {
  for(int i=0; i<n; ++i) {
    c[i] = a[i] + b[i];
  }
}


int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c       
	int errors=0;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	// Number of blocks must be enough for all N values without truncation
	dim3 blocksPerGrid((unsigned int)ceil(N / (double)THREADS_PER_BLOCK), 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// perform cpu version
	vectorAddCPU(N, a, b, c_ref);

	// check everything
	for(int i=0; i<N; ++i) {
	  if( (c[i] - c_ref[i])!=0) {
	    std::cout << "Error: mismatch for element " << i << " cpu = " << c_ref[i] << " gpu = " << c[i] << std::endl;
	    ++errors;
	  }
	}
	std::cout << "N(errors) = " << errors << std::endl;

	// Cleanup
	free(a); free(b); free(c); free(c_ref);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}
