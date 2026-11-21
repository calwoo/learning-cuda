# Week 2: Triton programs and vectorized kernels

[Back to the curriculum](../README.md) · [Week 1](../week1/README.md)

**Goal:** Use one Triton program per tile to implement and test vector addition and a fused affine transform. Explain how that mapping compares with your CUDA vector addition.

## 1. Learn the Triton launch model

1. Watch GPU MODE lecture 14, *Practitioner's Guide to Triton*. [GPU MODE lectures](https://github.com/gpu-mode/lectures)
2. Work through the vector-add tutorial. Trace how the launch grid picks a number of programs, `tl.program_id(0)` selects a tile, `tl.arange` constructs offsets inside it, and `tl.load` and `tl.store` use masks for the tail. [Triton Tutorial 1](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
3. For `N = 1025` and `BLOCK_SIZE = 256`, work out the program count and the valid offsets in the last tile. Compare this with the thread and block indices in your Week 1 kernel.

**Check:** Five programs cover the vector. The last program has one valid element at index 1,024.

## 2. Complete Triton Puzzles 1–2

Open the upstream notebook in Colab and run its setup cells. The notebook uses an interpreter for puzzle checks and includes a visualization dependency. The standalone checks below run directly in the local `cuda` environment. [Triton Puzzles notebook](https://github.com/gpu-mode/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb) · [Open in Colab](https://colab.research.google.com/github/gpu-mode/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb)

- [ ] **1. Constant Add.** Load, add, and store a vector that fits exactly into one tile.
- [ ] **2. Constant Add Block.** Distribute the work over multiple programs. Mask the last tile's loads and stores to cover only valid offsets.

## 3. Implement and test two kernels

1. Fill in [vector_add.py](vector_add.py) to compute `out = a + b` for contiguous 1D tensors. The runner sets the grid from the vector length and block size. Use `tl.program_id` to find this program's tile, then mask loads and stores beyond the end.
2. Fill in [affine.py](affine.py) to compute `out = alpha * x + b` in one kernel, using a scalar `alpha` and a per-element bias vector `b`.
3. Run [check.py](check.py) against the PyTorch references. It checks block sizes 128, 256, 512, and 1,024 with tiny inputs, partial tiles, exact multiples, and a larger case.

Run these commands in PowerShell from the repository root. The local `cuda` environment uses CPU PyTorch 2.14.0 and `triton-windows` 3.8.0.post29 for these checks. [Triton Windows setup](https://github.com/triton-lang/triton-windows)

```powershell
conda activate cuda
cd gpu-programming-lab/week2
$env:TRITON_INTERPRET = '1'
python -B check.py vector_add
python -B check.py affine
```

Both checks should fail until you fill in the kernel bodies. In a compatible Linux GPU environment with CUDA-enabled PyTorch and Triton, clear `TRITON_INTERPRET` and pass `--device cuda` to run the same checks on the GPU.

## 4. Record results

- [ ] Save both kernels with run commands and correctness results.
- [ ] On a supported GPU, compare tile sizes and benchmark the fused kernel against the equivalent eager PyTorch expression. Warm up each configuration and measure the operation without compilation or host transfers.
- [ ] Explain how a Triton program instance relates to a CUDA thread block on NVIDIA. Describe the difference between elements in a Triton tile and threads executing it, and identify which loads and stores the tail mask suppresses.
