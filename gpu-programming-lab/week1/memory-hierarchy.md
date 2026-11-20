# GPU memory hierarchy

[Week 1 guide](README.md)

In [vector addition](vector_add.cu), the CPU initializes `a` and `b` in host memory. `cudaMalloc` allocates `d_a`, `d_b`, and `d_c` in GPU global memory. `cudaMemcpy` moves the inputs to the GPU and the result back to the host. During the kernel, each thread loads two elements, adds them, and stores its result. [CUDA Programming Guide · memory hierarchy](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#memory-hierarchy)

| Level or space | Who uses it? | What to look for |
| --- | --- | --- |
| Registers | Private to a thread, allocated from its SM's register file | The compiler can hold the index and loaded operands in registers. |
| Shared memory | Threads in the same block for this lab | Explicit scratch space on the SM, declared with `__shared__`. A later tiled kernel can stage values here for reuse. |
| L1 cache | Warps running on one SM | Hardware-managed cache. On Turing, L1 and shared memory share part of the same on-chip storage. |
| L2 cache | SMs across the GPU | Hardware-managed cache that can serve global-memory traffic before it reaches device DRAM. |
| Global memory | Threads across the GPU | Device allocations live in this address space, backed by device DRAM. The GTX 1660 Ti uses 6 GB of GDDR6. |
| Host RAM | The CPU | Holds the input and result vectors on the host side of the copies. |

Sources for the hardware details are the [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) and [NVIDIA GeForce specifications](https://www.nvidia.com/en-eu/geforce/graphics-cards/compare/).

## Follow the loads

A load from `d_a[index]` addresses global memory. A cache hit can serve the load on chip. The loaded value can then live in a register while the thread computes the sum. **Shared memory is a separate address space** and is used when the kernel explicitly stages data there. Vector addition reads each input element once per launch and has no cross-thread reuse for shared memory to capture.

For a full warp in the exercise, consecutive lanes read consecutive `float` elements from each input and write consecutive output elements. The memory hardware can combine nearby lane requests into fewer transactions, a behavior called **coalescing**. Also distinguish CUDA's **local memory** from registers. Local memory is thread-private in scope and backed by device memory. Register spills are one reason a compiler might use it. [CUDA Programming Guide · device memory accesses](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#device-memory-accesses)

**Check:** Draw the path from `a[i]` on the host to a CUDA thread's operand and back through the output buffer. Mark which transfers your program requests and which parts the GPU caches handle automatically.
