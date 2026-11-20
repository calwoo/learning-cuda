# Week 1: Your first GPU kernels

[Back to the curriculum](../README.md)

**Goal:** Write a correct CUDA C++ vector addition kernel and explain how threads, warps, blocks, and streaming multiprocessors (SMs) execute it. Use that model to prepare for Triton in [Week 2](../week2/README.md).

## Study notes

- [Warp counting](warp-counting.md) works through the vector-add launch and its final partial block.
- [SIMT versus SIMD](simt-vs-simd.md) compares CUDA threads with CPU vector instructions and explains divergence.
- [SM scheduling](sm-scheduling.md) shows how a scheduler uses ready warps to hide latency.
- [Memory hierarchy](memory-hierarchy.md) traces vector data from host memory through the GPU.

## Before starting

You need access to an NVIDIA GPU, a compatible driver, the CUDA toolkit (`nvcc`), and a C++ compiler supported by that toolkit. A cloud GPU or a CUDA-enabled notebook is fine. The puzzles use Python and Numba; follow their repository's setup instructions. Verify your environment with `nvidia-smi` and `nvcc --version`, and record the GPU, driver, toolkit, and compiler versions with your results. If using a hosted notebook, save your work back to this repository.

## 1. Learn the execution model

1. Read NVIDIA's [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/). Trace the path from host allocation through kernel execution to result checking.
2. Read *Programming Massively Parallel Processors*, fourth edition, chapter 2, sections 2.1–2.7. Focus on host and device memory, kernels, threads, blocks, and grids.
3. Read chapter 4, sections 4.1–4.6, for SIMT execution, warps, SMs, and the scheduler.
4. Study Stanford CS149 lecture 7, *GPU Architecture and CUDA Programming*. [Stanford CS149 lectures](https://gfxcourses.stanford.edu/cs149/fall25/lecture/)
5. In your own notes, sketch a host program that allocates inputs on the device, copies data, launches a kernel, copies the result back, and releases memory.

**Check:** Explain the difference between host and device code, what a kernel launch does, and why data must be available on the device before the kernel reads it.

## 2. Watch and trace

Watch GPU MODE lecture 2, the recap of PMPP chapters 1–3, using the [lecture materials](https://github.com/gpu-mode/lectures/tree/main/lecture_002) and the [GPU MODE channel](https://www.youtube.com/@GPUMODE). Pause to trace a one-dimensional launch by hand. For 128 threads per block and 1,000 elements, calculate the number of blocks, the global index of thread 5 in block 2, and how many launched threads have no corresponding element.

**Check:** The launch needs 8 blocks, that thread has index 261, and 24 threads are beyond the end. Explain why those threads must not access the arrays.

For a second launch, predict how many blocks and warps cover 100,000 elements with 256 threads per block and 32 threads per warp. Check your result against 391 blocks and 3,128 warps. Explain what happens to the extra threads.

Compare GPU SIMT and CPU SIMD in your notes. Explain what a warp scheduler can do when one warp is waiting on a memory load. Sketch registers and shared memory/L1 on an SM, then L2 and device DRAM (such as HBM on data center GPUs). Mark where the vector inputs live and which values a thread might hold in registers.

## 3. Practice indexing with GPU Puzzles

Work through [GPU Puzzles](https://github.com/srush/GPU-Puzzles) in order:

- [ ] **1: Map** — map one thread to each output element.
- [ ] **2: Zip** — combine corresponding elements of two inputs.
- [ ] **3: Guards** — protect accesses when the launch covers more threads than elements.

Run the provided checks. For each puzzle, write down the global index calculation and the condition under which a thread should write. Try to solve each before consulting a solution. These exercises use Numba's CUDA model; the final exercise below uses CUDA C++.

## 4. Build vector addition

Create `vector_add.cu` in this directory. Implement `c[i] = a[i] + b[i]` for single-precision vectors:

1. Initialize nontrivial inputs on the host and calculate a CPU reference.
2. Allocate device arrays and copy both inputs to the GPU.
3. Launch a kernel with `i = blockIdx.x * blockDim.x + threadIdx.x`; guard reads and writes with `i < N`.
4. Copy the result back, compare every element to the CPU reference with a suitable floating-point tolerance, and return a failing exit code on mismatch.
5. Check CUDA API and launch errors, synchronize before reporting kernel errors, and free device allocations.

Compile and run using your environment's CUDA toolchain, for example:

```bash
nvcc -O2 -o vector_add vector_add.cu
./vector_add
```

Allow the vector length and threads per block to be changed, through command-line arguments or a clearly documented test loop. For positive `N`, use ceiling division `(N + threads_per_block - 1) / threads_per_block` for the grid size.

## 5. Test, measure, and explain

- [ ] Test every combination of `N = 1, 32, 33, 1024, 1025, 1000000` and 1, 32, 128, and 256 threads per block, subject to the device's supported launch limits. Record the grid size and whether each case passes.
- [ ] Compare at least several sizes using CUDA events around the kernel, with warm-up runs and synchronization before reading elapsed time. Record the timing method and distinguish kernel time from allocation and transfer time. Treat these measurements as a baseline, not a speed target.
- [ ] Explain why the last block can contain idle threads and why the guard is required even when most blocks are full.
- [ ] Describe the thread-to-element mapping and contiguous memory accesses. Note the GPU and software versions used.

## Deliverables and completion check

Keep the source, instructions for building and running it, correctness results, a small timing table, and your explanations in this folder. Before starting week 2, make sure you can answer without looking up an example:

- What do `threadIdx.x`, `blockIdx.x`, `blockDim.x`, and `gridDim.x` represent?
- How do you compute the global index and the required number of blocks?
- What happens when `N` is not divisible by the block size, and how does the bounds check prevent an invalid access?
- Which operations run on the CPU and GPU, and when does data move between them?
- How does a block break into warps, and how can an SM hide memory latency by scheduling a ready warp?
- Which parts of the memory hierarchy are local to an SM, and which parts serve the whole GPU?
