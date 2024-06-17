#include <cuda.h>
#include <stdio.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    // initialize device memory
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // run kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // move output from device
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int n = 1000;
    float A_h[n];
    float B_h[n];
    float C_h[n];

    // generate some dummy vectors to add
    for (int i = 0; i < n; i += 1) {
        A_h[i] = float(i);
        B_h[i] = A_h[i] / 1000.0f;
    }

    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < n; i += 1) {
        if (i > 0) {
            printf(", ");
            if (i % 10 == 0) {
                printf("\n");
            }
        }
        printf("%8.3f", C_h[i]);
    }
    printf("\n");
}