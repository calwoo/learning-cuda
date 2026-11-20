# SIMT versus SIMD

[Week 1 guide](README.md)

**SIMD** means *single instruction, multiple data*. A CPU vector instruction works on packed values. For example, an AVX 256-bit floating-point add takes two vectors of eight `float` values and produces eight sums. A compiler can also generate vector instructions from a scalar loop when the dependencies permit it. [Intel instruction reference · packed floating-point add](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_add_ps)

**SIMT** means *single instruction, multiple threads*. In CUDA, you write the operation for one thread. Each thread has its own index and state. The hardware groups threads into warps and issues an instruction to the lanes participating in that instruction. [CUDA Programming Guide · SIMT architecture](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#simt-architecture)

| Question | CPU SIMD example | CUDA SIMT example |
| --- | --- | --- |
| What does the program express? | Add two groups of eight floats. | Have each thread add one pair of floats. |
| Where does the data index come from? | The loop picks the start of a vector, and each lane corresponds to a position in it. | Each thread computes its index using its block and thread IDs. |
| How do pieces of the input take different paths? | The vector code can use masks, select between results, or handle part of the work separately. | Threads can take different paths, and instructions on each path operate on the lanes participating in it. |

## Divergence in vector addition

The body of `if (index < n)` is the same for every valid lane. In the first warp of the last block for `n = 1025`, just lane 0 does the addition. Most warps in the launch have all 32 lanes doing the same work.

Now imagine a kernel with a different calculation for even and odd indices. Every full warp contains both kinds of index. If the compiler emits a branch, that warp needs to execute instructions on both paths with different groups of lanes. For short branches, the compiler may use predication, which makes individual instructions conditional on a per-lane flag. Different warps can take different paths independently. [CUDA Programming Guide · control flow](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#control-flow-instructions)

On Turing, Independent Thread Scheduling lets threads within a warp diverge and reconverge at finer granularity. When threads exchange data, use the appropriate synchronization for the participating threads. Vector addition needs no such exchange because each thread owns a distinct output. [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

**Check:** Compare branching on `index % 2 == 0` with branching on `blockIdx.x % 2 == 0`. Which condition is uniform within a warp? Why?
