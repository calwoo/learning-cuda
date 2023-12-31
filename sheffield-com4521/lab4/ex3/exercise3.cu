#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 512
#define M 256
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);



__global__ void matrixAdd(int *a, int *b, int *c, int numRows, int numCols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < numRows) && (j < numCols)) {
        c[i*numCols+j] = a[i*numCols+j] + b[i*numCols+j];
    }
}


void matrixAddCPU(int *a, int *b, int *c, int numRows, int numCols) {
	for (int i = 0; i < numRows * numCols; i++) {
		c[i] = a[i] + b[i];
	}
}

int validate(int *c, int *c_ref, int numRows, int numCols) {
	int numErrors = 0;
	for (int i = 0; i < numRows * numCols; i++) {
		if (c[i] != c_ref[i]) {
			numErrors += 1;
			printf("error found: %d != %d\n", c[i], c_ref[i]);
		}
	}
	return numErrors;
}



int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int);

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
    dim3 blocksPerGrid(N/16, M/16, 1);
    dim3 threadsPerBlock(16, 16, 1);
	matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, M);
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	matrixAddCPU(a, b, c_ref, N, M);

	errors = validate(c, c_ref, N, M);
	printf("num errors: %d\n", errors);

	// Cleanup
	free(a); free(b); free(c);
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
	for (unsigned int i = 0; i < N * M; i++){
		a[i] = rand();
	}
}
