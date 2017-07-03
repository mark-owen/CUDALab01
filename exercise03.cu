// -*-c-*-
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define NR 2048
#define NC 2048
#define THREADS_PER_BLOCK 256

void checkCUDAError(const char*);
void random_ints(int *a);



__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < max) c[i] = a[i] + b[i];
}

__global__ void matrixAdd(int *a, int *b, int *c, int max) {
  // version for 1D block of threads
  //int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if( i < max ) c[i] = a[i] + b[i];

  // version for 2D block of threads
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int loc = i +  NR * j;
  if(i*j < max) c[loc] = a[loc] + b[loc];
}

void vectorAddCPU(int n, int* a, int*b, int*c) {
  for(int i=0; i<n; ++i) {
    c[i] = a[i] + b[i];
  }
}

void matrixAddCPU(int n, int m, int* ma, int* mb, int* mc) {
  for(int i=0; i<n; ++i) {
    for(int j=0; j<m; ++j) {
      int loc = i +  n * j;
      mc[loc] = ma[loc] + mb[loc];
    }
  }
}


int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c       
	int errors=0;
	unsigned int size = NR * NC * sizeof(int);

	// Alloc space for device copies of a, b, c	
	//cudaMalloc((void **)&d_a, sizeof(int *));
	//for(int i=0; i<NR; ++i) cudaMalloc((void **)&(d_a[i]), NC*sizeof(int));

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
	dim3 blocksPerGrid((unsigned int)ceil(NC*NR / (double)THREADS_PER_BLOCK), 1, 1);
	//dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	dim3 threadsPerBlock(sqrt(THREADS_PER_BLOCK), sqrt(THREADS_PER_BLOCK), 1);
	
	//dim3 blocksPerGrid(1, 1, 1);
	//dim3 threadsPerBlock(NR, NC, 1);
	matrixAdd<< <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, NR*NC);
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// perform cpu version
	matrixAddCPU(NR, NC, a, b, c_ref);

	// check everything
	for(int i=0; i<NR*NC; ++i) {
	  if( (c_ref[i] - c[i])!=0 ) {
	    std::cout << "ERROR at element " << i << std::endl;
	    ++errors;
	  }
	}

	// Cleanup
	free(a); free(b); free(c); free(c_ref);
	//cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	//checkCUDAError("CUDA cleanup");

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
	for (unsigned int i = 0; i < NR*NC; i++){
		a[i] = rand();
	}
}
