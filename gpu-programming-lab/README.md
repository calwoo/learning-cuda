# GPU Programming Lab

A sequential, hands-on path from CUDA fundamentals to GPU kernels for LLM inference, based on the [GPU Programming Roadmap](chatgpt-conversation://6afb8d79-6b58-83e9-a714-a87db5bc46e5). The expanded roadmap targets 20 weeks at roughly 8–10 hours per week (160–200 hours); use the completion criteria, rather than the calendar, to set the pace.

## How to work

Each week, budget roughly two hours for reading, one to two for lectures, three to four for exercises, and two for a project or benchmark. For each exercise, keep the kernel, correctness tests against PyTorch or NumPy, benchmarks across tensor sizes, and a short explanation of thread mapping and memory access. For optimizations, compare versions and explain measured improvements or regressions. Record hardware, software versions, shapes, dtypes, and timing method so results are reproducible.

## Curriculum

### 1. CUDA fundamentals (weeks 1–3)

- **Week 1:** Study CUDA's execution model and PMPP chapter 2; complete GPU Puzzles 1–3 and write and test CUDA C++ vector addition across boundary sizes and block sizes.
- **Week 2:** Study multidimensional grids and layout (PMPP chapter 3, GPU MODE lecture 3); complete puzzles 4–7, rectangular matrix addition, and a 3×3 blur with boundary handling.
- **Week 3:** Study warps, divergence, scheduling, and grid-stride loops (PMPP chapter 4, GPU MODE lectures 4–5). Experiment with block sizes and branching. **Milestone:** implement and benchmark a fused `max(0, alpha * x + b)` CUDA operator against PyTorch.

### 2. Memory and parallel algorithms (weeks 4–7)

- **Week 4:** Study coalescing, shared memory, and bank conflicts (PMPP chapter 5); compare naive, tiled, and padded transpose against a copy kernel.
- **Week 5:** Study data reuse and performance (PMPP chapter 6); implement naive, shared-memory tiled, and register-tiled GEMM. **Milestone:** compare correctness and performance with PyTorch and cuBLAS across square and rectangular shapes.
- **Week 6:** Study reductions, histograms, and atomics (PMPP chapters 9–10, GPU MODE lecture 9); implement shared-memory and warp-shuffle reductions, a multi-block reduction, and histograms with different aggregation strategies.
- **Week 7:** Study scans (PMPP chapter 11, GPU MODE lectures 20–21); implement Kogge–Stone and work-efficient scans, a multi-block scan, and stream compaction. **Milestone:** package GEMM, reduction, histogram, and scan with tests and benchmarks, including irregular sizes.

### 3. Architecture, profiling, and debugging (weeks 8–10)

- **Week 8:** Study SM scheduling, registers, shared memory, and occupancy; vary resource use and compare predicted and measured occupancy.
- **Week 9:** Use Nsight Systems and Nsight Compute to investigate GEMM. Write a report with at least three performance hypotheses, experiments, and conclusions.
- **Week 10:** Use Compute Sanitizer to investigate memory and synchronization errors; integrate a kernel with PyTorch and test registration with `torch.library.opcheck`. **Milestone:** turn the fused operator into a tested extension with repeatable CUDA-event benchmarks and a breakdown of device and host overhead.

### 4. Triton and ML kernels (weeks 11–14)

- **Week 11:** Work through Triton vector addition and fused softmax; implement both, including masking for irregular sizes.
- **Week 12:** Work through Triton matrix multiplication; tune tiles and warp counts, test supported dtypes, and compare with CUDA GEMM and PyTorch.
- **Week 13:** Study normalization and fusion; implement RMSNorm and fused bias, activation, and residual. **Milestone:** a tested and benchmarked Triton library containing softmax, RMSNorm, and fused operations.
- **Week 14:** Study online softmax and FlashAttention; implement tiled causal attention forward in Triton. **Milestone:** compare numerical accuracy, latency, and temporary memory against naive attention without materializing the full score matrix.

### 5. Advanced CUDA (weeks 15–17)

- Study Tensor Cores, warp-level matrix operations, and the CUDA WMMA GEMM sample. Compare supported precisions and numerical error, and inspect generated instructions.
- Study CUTLASS, CuTe, asynchronous data movement, tiling, and pipelining. Explore how matrix tiles and data movement map onto the hardware, then benchmark an advanced GEMM implementation on compatible hardware.
- Study streams, events, CUDA graphs, and integration overhead; use measurements to explain when launch and synchronization costs matter.

### 6. LLM inference (weeks 18–20)

- Compare prefill and decode attention, their memory-access patterns, and their performance constraints. Implement and benchmark a single-query decode attention kernel.
- Study paged KV caches, quantization, and sampling in inference engines such as vLLM and SGLang. Explore a paged-attention or quantized kernel, with correctness tests against a reference.
- **Capstone:** profile an inference pipeline, identify a measured bottleneck, improve a relevant kernel or execution path, and report correctness, latency, and the limits of the result.

The final phases are project-oriented and may take longer than their nominal weeks. Advanced features and profiling counters depend on the GPU and execution environment.

## Core resources

- [Programming Massively Parallel Processors, fourth edition](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/) (PMPP): main textbook.
- [GPU MODE lectures and code](https://github.com/gpu-mode/lectures), with videos on the [GPU MODE channel](https://www.youtube.com/@GPUMODE).
- [GPU Puzzles](https://github.com/srush/GPU-Puzzles) and [Stanford CS149 lectures](https://gfxcourses.stanford.edu/cs149/fall25/lecture/).
- [CUDA introduction](https://developer.nvidia.com/blog/even-easier-introduction-cuda/), [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/), and [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/), the [FlashAttention paper](https://arxiv.org/abs/2205.14135), [CUDA samples](https://github.com/NVIDIA/cuda-samples), and [CUTLASS](https://github.com/NVIDIA/cutlass).

## Suggested organization

Create these directories as work begins: `01-cuda-fundamentals/`, `02-memory-algorithms/`, `03-profiling/`, `04-triton/`, `05-advanced-cuda/`, `06-llm-inference/`, and `benchmarks/`. Existing repository exercises can be reused where useful.
