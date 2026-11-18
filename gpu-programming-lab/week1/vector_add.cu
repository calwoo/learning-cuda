// Week 1 scaffold: add two vectors on the GPU.
// Build: nvcc -O2 -o vector_add vector_add.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

// TODO: Handle CUDA API errors and report the failed operation.

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Calculate this thread's global index.
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    // If the index is in bounds, add the input elements into c.
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main(int argc, char** argv) {
    // TODO: Read n and threads_per_block from the command line (or choose defaults).
    // Validate both before calculating the grid size.
    const int n = 1025;
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Initialize host inputs with nontrivial values and compute a CPU reference.
    std::vector<float> a(n), b(n), result(n), expected(n);

    // Allocate three device arrays and copy both inputs to the device.
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const size_t bytes = n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // TODO: Launch vector_add<<<blocks, threads_per_block>>>(...).
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    // TODO: Check launch and synchronization errors.
    cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // TODO: Compare every result against the CPU reference using a tolerance.
    // Print a useful failure message and return a nonzero exit code on failure.
    float tol = 0.0001;
    for (int i = 0; i < n; i++) {
        expected[i] = a[i] + b[i];
        if (fabs(result[i] - expected[i]) >= tol) {
            std::cout << "exceeded error" << std::endl;
        }
    }


    // TODO: Free device allocations, including on error paths.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
