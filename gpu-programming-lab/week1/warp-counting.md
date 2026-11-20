# Warp counting

[Week 1 guide](README.md)

On NVIDIA GPUs, a **warp** is a group of 32 thread positions within one block. For a one-dimensional block, warp 0 contains threads 0–31, warp 1 contains 32–63, and so on. A block of 48 threads has two warps. The second warp has 16 threads and 16 unused lane positions. Warp boundaries restart at every block. [CUDA Programming Guide · SIMT architecture](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#simt-architecture)

## Count blocks first, then warps

For the Week 1 kernel, one thread processes one element. Round up twice, once to cover the input and once **per block** to count its warps.

| Quantity | Calculation |
| --- | --- |
| Blocks in the grid | `ceil(N / threads_per_block)` |
| Warps in one block | `ceil(threads_per_block / 32)` |
| Warps across the grid | `blocks × warps_per_block` |
| CUDA threads launched | `blocks × threads_per_block` |
| Threads outside the input | `threads_launched − N` |

With 100,000 elements and 256 threads per block, the grid has **391 blocks**. Each block has eight warps, for **3,128 warps** and 100,096 threads across the launch. The first 390 blocks cover 99,840 elements. The last block covers the remaining 160 elements with its first five warps. The other three warps have no valid element, so their threads skip the body of `if (index < n)`. Those threads still need to evaluate the guard.

## Trace the actual exercise

For `n = 1025` and 256 threads per block, the [vector-add kernel](vector_add.cu) launches five blocks and 40 warps. Blocks 0–3 cover indices 0–1,023. In block 4, only thread 0 has a valid index. Its warp has one lane doing the addition, and the other seven warps in that block each have zero lanes doing it.

The launch count describes the whole grid. Only a subset of those warps can be resident at any instant. [SM scheduling](sm-scheduling.md)

**Check:** Three blocks of 48 threads give six warps, even though rounding 144 threads up to groups of 32 would give five. Explain why the 16 unused lane positions in one block cannot hold threads from another block.
