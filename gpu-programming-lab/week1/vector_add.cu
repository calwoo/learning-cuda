// Week 1 scaffold: add two vectors on the GPU.
// Build: nvcc -O2 -o vector_add vector_add.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TODO: Handle CUDA API errors and report the failed operation.

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: Calculate this thread's global index.
    // TODO: If the index is in bounds, add the input elements into c.
}

int main(int argc, char** argv) {
    // TODO: Read n and threads_per_block from the command line (or choose defaults).
    // Validate both before calculating the grid size.
    const int n = 1025;
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // TODO: Initialize host inputs with nontrivial values and compute a CPU reference.
    std::vector<float> a(n), b(n), result(n), expected(n);

    // TODO: Allocate three device arrays and copy both inputs to the device.
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // TODO: Launch vector_add<<<blocks, threads_per_block>>>(...).
    // Check launch and synchronization errors, then copy the result to the host.

    // TODO: Compare every result against the CPU reference using a tolerance.
    // Print a useful failure message and return a nonzero exit code on failure.

    // TODO: Free device allocations, including on error paths.
    return 0;
}
