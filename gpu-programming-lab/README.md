# GPU Programming Lab

This 20-week curriculum starts with one CUDA C++ kernel, then uses Triton from Week 2 onward to learn parallel algorithms, GPU performance, and LLM inference. Gluon introduces explicit hardware control in the advanced labs. [GPU Programming Roadmap](chatgpt-conversation://6afb8d79-6b58-83e9-a714-a87db5bc46e5)

## How to work

Read the assigned material, derive the thread or tile mapping, implement the kernel, then test it against PyTorch or NumPy. Include irregular shapes and relevant dtypes. For benchmarks, record the GPU, software versions, shapes, kernel configuration, and timing method. Keep compilation and transfers outside kernel timing, and explain results using memory traffic, parallelism, or profiler evidence.

Use the lab's `cuda` conda environment for local CUDA and Numba work. Start Triton Puzzles through their CPU interpreter. For compiled Triton exercises and performance measurements, use a compatible Linux GPU environment. Upstream Triton lists NVIDIA compute capability 8.0 or newer, while the local GTX 1660 Ti is 7.5. Check the hardware required by each Gluon tutorial before running it. [Triton compatibility](https://github.com/triton-lang/triton#compatibility) · [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles)

## 1. GPU architecture and CUDA (Week 1)

### [Week 1 · Threads, warps, blocks, SMs, and memory](week1/README.md)

- **Study.** Read NVIDIA's [CUDA introduction](https://developer.nvidia.com/blog/even-easier-introduction-cuda/), PMPP chapter 2 §§2.1–2.7 and chapter 4 §§4.1–4.6, Stanford CS149 lecture 7, and GPU MODE lecture 2.
- **Practice.** Complete GPU Puzzles 1–3 and build and test CUDA C++ vector addition. Predict the block and warp counts for 100,000 elements with 256 threads per block. Compare SIMT with CPU SIMD, draw the memory hierarchy, and explain how the SM hides memory latency.

## 2. Triton fundamentals (Weeks 2–4)

### [Week 2 · Triton programs and vectorized kernels](week2/README.md)

- **Study.** Watch GPU MODE lecture 14 and work through [Triton Tutorial 1](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html). Complete Triton Puzzles 1–2.
- **Practice.** Write masked vector addition and fused `y = alpha * x + b` with `triton.jit`. Compare tile sizes 128, 256, 512, and 1024 on lengths that include partial tiles. Explain how a Triton program instance relates to a CUDA thread block.

### Week 3 · Multidimensional tensors and addressing

- **Study.** Read PMPP §§3.1–3.3, revisit masking in Tutorial 1, and complete Triton Puzzles 3–6.
- **Practice.** Map a 2D launch grid onto rectangular matrix addition. Test a single row, a single column, and shapes where both dimensions leave partial tiles. Implement row-wise broadcasting and a 3×3 image blur in Triton. Extend one kernel to accept explicit input strides and test transposed and sliced views. Derive the grid, pointer offsets, and masks from a given shape and strides.

### Week 4 · Coalescing, tiling, and bandwidth

- **Study.** Read PMPP §§5.1–5.5 and §6.1 and NVIDIA's [matrix transpose article](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).
- **Practice.** Implement a tiled transpose in Triton. Vary tile dimensions, compare against contiguous copy, and use effective bandwidth to explain how the access patterns affect performance.
- **Milestone 1.** A fundamentals library with vector operations, broadcasting, a 2D stencil, and transpose. Show correctness for irregular shapes and compare latency and bandwidth.

## 3. Parallel algorithms through Triton (Weeks 5–8)

### Week 5 · Reductions

- **Study.** Read PMPP §§10.1–10.8, watch GPU MODE lecture 9, and solve Triton Puzzle 7 (*Long Sum*).
- **Practice.** Derive tree reduction work and depth. Use `tl.sum` within one tile, then reduce partial results across programs in a second stage. Implement dot product and row-wise maximum. Compare one-stage and multistage reductions and explain where communication occurs.

### Week 6 · Prefix scans

- **Study.** Read PMPP §§11.1–11.7, watch GPU MODE lectures 20–21, and review Stanford CS149 lecture 8.
- **Practice.** Compare Kogge–Stone and Brent–Kung work and depth. Implement inclusive scan with `tl.associative_scan`, scan across tiles using tile totals, and use the prefix sums for stream compaction. Explain how a tile gets its global offset.

### Week 7 · Histograms, atomics, and contention

- **Study.** Read PMPP §§9.1–9.6, review GPU MODE's reduction materials, and look up [`tl.histogram`](https://triton-lang.org/main/python-api/generated/triton.language.histogram.html) and [`tl.atomic_add`](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html).
- **Practice.** Compare a global-atomic histogram with per-tile histograms and a second aggregation kernel. Measure uniform and skewed inputs to explain the impact of contention.

### Week 8 · Parallel matrix multiplication

- **Study.** Read PMPP §§5.3–5.6 and §§6.2–6.5 and work through [Triton Tutorial 3](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
- **Practice.** Build a baseline tiled GEMM using explicit products and reduction. Then implement a blocked GEMM using `tl.dot`. Tune the M, N, and K tile sizes, estimate reuse and traffic, and compare with `torch.matmul` on rectangular shapes.
- **Milestone 2.** A library with reduction, scan, histogram, and GEMM. Include correctness tests, work and depth analysis where applicable, and benchmarks.

## 4. Profiling and performance engineering (Weeks 9–11)

### Week 9 · Performance models

- **Study.** Read PMPP §4.7 and chapter 6 and watch GPU MODE lectures 4 and 8.
- **Practice.** Estimate memory traffic and arithmetic intensity for GEMM and reduction, then construct roofline predictions. Vary `num_warps` and tile sizes and explain the effect of register pressure and occupancy.

### Week 10 · Profiling Triton kernels

- **Study.** Watch GPU MODE lecture 16 and review [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), and [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).
- **Practice.** Profile transpose, reduction, and GEMM. Use the timeline and available occupancy, throughput, and stall counters to distinguish launch, bandwidth, and compute limits. Compare observations with Week 9 predictions.

### Week 11 · Compiler inspection and debugging

- **Study.** Watch GPU MODE lectures 29 and 37 and read [Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
- **Practice.** Inspect Triton IR, LLVM IR, PTX, and SASS where available. Trace a reduction through lowering and identify memory operations and resource use. Diagnose a bad offset or mask with the interpreter and use Compute Sanitizer on the GPU path.
- **Milestone 3.** Test three optimization hypotheses for each of two earlier kernels. Write a short report relating measured changes to hardware evidence.

## 5. ML kernels in Triton (Weeks 12–14)

### Week 12 · Online softmax

- **Study.** Work through [Triton Tutorial 2](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), solve Triton Puzzle 8 (*Long Softmax*), and review hierarchical reductions.
- **Practice.** Implement stable row-wise softmax. Derive the running maximum and denominator updates, then implement a streaming version for long rows. Compare numerical error and memory traffic.

### Week 13 · Normalization and fusion

- **Study.** Work through [Triton Tutorial 5](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html), then study GPU MODE lecture 18 and the RMSNorm notebook from lecture 28.
- **Practice.** Implement RMSNorm, fused bias + GELU, and fused residual addition + RMSNorm. Compare with equivalent PyTorch operations and examine accumulation precision for FP32 and BF16 inputs.
- **Milestone 4.** Tested softmax, RMSNorm, and activation operators. Show when fusion reduces intermediate storage or latency.

### Week 14 · FlashAttention

- **Study.** Read the [FlashAttention paper](https://arxiv.org/abs/2205.14135), watch GPU MODE lecture 12, solve Triton Puzzle 9, and study the forward pass of [Triton Tutorial 6](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
- **Practice.** Derive the running maximum, denominator, and weighted-value accumulator. Implement naive PyTorch attention as a reference, then tiled Triton attention with causal masking.
- **Milestone 5.** Exact forward attention that avoids storing the full score matrix in device memory. Compare numerical error, latency, and temporary memory against the reference.

## 6. Advanced Triton and Gluon (Weeks 15–17)

### Week 15 · Explicit layouts

- **Study.** Watch GPU MODE lecture 104 and read Gluon's [introduction](https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html) and [tensor layouts tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html).
- **Practice.** Rewrite vector addition and revisit transpose with explicit layouts. Use `BlockedLayout`, `SliceLayout`, and layout conversion. Draw how registers, lanes, and warps own elements, then compare layouts for coalescing and reduction communication.

### Week 16 · Tensor Cores and asynchronous pipelines

- **Study.** Watch GPU MODE lecture 23, revisit Triton Tutorial 3, and choose the async copy, TMA, and warp-group MMA material that fits your hardware. [Gluon tutorials](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html)
- **Practice.** Inspect how `tl.dot` lowers to matrix instructions. Compare available FP32, TF32, FP16, and BF16 paths, including numerical error. Adapt an asynchronous copy pipeline and compare single buffering with different pipeline depths.

### Week 17 · Warp specialization and persistent kernels

- **Study.** Read [Warp Specialization in Triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) and Gluon's [warp specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html) and [persistence](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html) tutorials.
- **Practice.** Trace producer and consumer warp roles. Adapt a persistent GEMM supported by your device, measure how pipeline depth affects resource use and throughput, and compare with Week 8. Use Hopper or newer hardware for Gluon's warp specialization exercises, or study the schedule using the tutorial code if compatible hardware is unavailable.
- **Milestone 6.** A hardware-aware GEMM comparison. Explain tile ownership, register and shared memory use, matrix instructions, and pipeline behavior.

## 7. LLM inference kernels (Weeks 18–20)

### Week 18 · Decode attention

- **Study.** Read the [vLLM paged attention walkthrough](https://docs.vllm.ai/en/stable/design/paged_attention/), watch GPU MODE lecture 40, and review Week 14.
- **Practice.** Implement single-query attention with a contiguous KV cache and variable context lengths. Add split-KV execution and merge partial softmax results. Compare prefill and decode arithmetic intensity and benchmark across batch sizes.

### Week 19 · Paged KV caches and quantization

- **Study.** Read [FlashInfer's KV-cache layout guide](https://docs.flashinfer.ai/tutorials/kv_layout.html), watch GPU MODE lecture 34, and solve Triton Puzzle 12 (*Quantized Matrix Mult*).
- **Practice.** Implement logical-to-physical page mapping and paged attention reads. Test randomized page tables and partial pages. Add quantized KV storage with scales and measure bandwidth and numerical error.
- **Milestone 7.** Paged decode attention tested across variable context lengths and page layouts, with latency measurements across batches and contexts.

### Week 20 · Decode pipeline capstone

- **Study.** Watch GPU MODE lecture 35 and review vLLM attention backend interfaces and your profiling reports.
- **Build.** Assemble one transformer block in decode mode with RMSNorm, RoPE, KV-cache append, paged attention, and a fused activation. Use library GEMM for the projections. Compare batches 1, 4, 16, and 32 at context lengths 512, 2,048, and 8,192. Measure kernel time, step latency, and effective cache bandwidth.
- **Final milestone.** A tested kernel collection, a standalone decode pipeline, and a report identifying measured bottlenecks. Optionally integrate a kernel into vLLM or SGLang and benchmark a serving workload.

## Reference material

- [Programming Massively Parallel Processors, fourth edition](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/) (PMPP) and [Stanford CS149](https://gfxcourses.stanford.edu/cs149/fall25/lecture/).
- [GPU MODE lectures and code](https://github.com/gpu-mode/lectures), [GPU Puzzles](https://github.com/srush/GPU-Puzzles), and [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/), [Gluon tutorials](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html), and the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).

Keep exercises and results in their `weekN/` folder. Put benchmarks shared across multiple weeks in `benchmarks/`.
