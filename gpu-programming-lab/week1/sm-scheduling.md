# SM scheduling

[Week 1 guide](README.md)

A **streaming multiprocessor (SM)** is a GPU execution unit that hosts one or more CUDA thread blocks. When there is capacity, the GPU assigns a block to an SM. All threads in that block run on the same SM, grouped into warps. A block's resident warps have resources such as registers allocated on that SM. Another block can be admitted when the SM has room. [CUDA Programming Guide · hardware implementation](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#hardware-implementation)

## How warp scheduling hides latency

Consider two warps resident on an SM. Warp A issues a load from global memory. Its next instruction needs the loaded data, so it cannot yet issue. If warp B has an instruction ready and the execution resources are available, a scheduler can issue B's instruction while A waits.

| What happens | Effect |
| --- | --- |
| Warp A issues a load. | The memory request is in flight. |
| A reaches an instruction that uses the load result. | A waits on that dependency. |
| Warp B has a ready instruction. | The scheduler can issue B's instruction. |
| A's data returns. | A becomes eligible again once its other dependencies are clear. |

Resident warps keep their execution state on the SM, so switching which warp issues an instruction does not require saving and restoring that state the way an operating-system context switch does. This overlaps useful work with the wait. If every warp is waiting, the scheduler has nothing ready to issue. [CUDA Programming Guide · hardware multithreading](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-c-programming-guide/index.html#hardware-multithreading)

## Launched, resident, and ready

- **Launched warps** count the whole grid.
- **Resident warps** are assigned to an SM and have resources allocated there.
- **Ready warps** can issue their next instruction when a scheduler and execution resources are available.

On your Turing GPU, the architectural limit is 32 resident warps per SM. A 256-thread block consumes eight warps, so the warp limit alone allows up to four such blocks per SM. Registers, shared memory, and other resource limits can reduce that number. **Occupancy** compares resident warps with the hardware's maximum. More resident warps give the scheduler more chances to find work, and throughput still depends on what those warps are waiting for. [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

**Check:** If an SM has 16 resident warps and all 16 are waiting for data, how many can issue their next dependent instruction?
