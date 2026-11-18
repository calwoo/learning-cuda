# Week 2: Multidimensional indexing and memory layout

[Back to the curriculum](../README.md) · [Week 1](../week1/README.md)

**Goal:** Map a 2D CUDA grid onto a row-major matrix. Write two kernels that handle rectangular inputs and partial blocks.

## 1. Trace a 2D launch

1. Read chapter 3 of *Programming Massively Parallel Processors*, fourth edition. Focus on sections 3.1–3.3 for grid organization, thread mapping, and image blur. Read section 3.4 for a second example of mapping a thread to an output element. [Chapter 3](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/xhtml/Ch003_47-68_B9780323912310000045.xhtml)
2. Watch Jeremy Howard's *Getting Started with CUDA for Python Programmers* and work through the lecture notebook's image examples. Trace how a thread's coordinates locate an element in a flat buffer. [GPU MODE Lecture 3](https://www.youtube.com/watch?v=4sgKnKbR-WE) · [Lecture materials](https://github.com/gpu-mode/lectures/tree/main/lecture_003)
3. On paper, use CUDA `x` for the column and `y` for the row. For a matrix with 5 rows and 9 columns and a block of `dim3(4, 4)`, calculate the grid dimensions, the number of threads outside the matrix, and the flat offset for thread `(0, 0)` in block `(2, 1)`. Explain why the row stride depends on the number of columns.

**Check:** The grid is `dim3(3, 2)`, 51 of the 96 launched threads are outside the matrix, and the specified thread maps to row 4, column 8, at offset 44.

## 2. Complete GPU Puzzles 4–7

Open the local notebook with the **Python (cuda)** kernel. Use each puzzle's axis and array conventions. Inspect the mapping with `problem.show()` and run `problem.check()` after each solution. [GPU Puzzles notebook](../GPU-Puzzles/GPU_puzzlers.ipynb)

- [ ] **4. Map 2D.** Map threads to a square matrix and guard both coordinates.
- [ ] **5. Broadcast.** Work out which element of each input contributes to each output cell.
- [ ] **6. Blocks.** Use the block and thread indices to compute a global index across multiple blocks.
- [ ] **7. Blocks 2D.** Extend the multi-block mapping to both axes and guard partial blocks.

After Puzzle 7, explain what changes when the block moves one position along each axis.

## 3. Implement rectangular matrix addition

Create `matrix_add.cu`. Reuse the allocation, copy, launch, and verification flow from Week 1.

1. Store two equally sized `height × width` matrices in flat row-major arrays. Initialize them with distinct values that vary across both rows and columns so an indexing mistake shows up in the output.
2. Start with a `16 × 16` block. Compute the grid by rounding up the width and height independently. Map one thread to each output element and check both bounds before accessing memory.
3. Derive the flat offset from the row, column, and width. Use that offset to add the input elements and write the sum.
4. Compare every output with a CPU reference. Report the first mismatch by row and column and return a failure status.

**Test:** Use shapes `1 × 1`, `1 × 19`, `19 × 1`, `16 × 32`, and `17 × 31` (rows × columns). Explain which cases launch partial blocks.

## 4. Implement a 3×3 box blur

Create `image_blur.cu` for a single-channel image stored as a flat row-major array. Reuse the matrix addition launch and thread mapping.

1. Give each thread one output pixel. For each valid output coordinate, walk its 3×3 neighborhood in the input.
2. Average only the neighbors inside the image. Divide by the number of samples included so the same rule works for interiors, edges, corners, and thin images.
3. Write to a separate output array so every thread reads the original image. Compare the result against a CPU implementation using the same edge rule.
4. Verify a constant image stays constant. For a 3×3 input containing 1 through 9 in row-major order, check that the center becomes 5 and the upper-left corner becomes 3. Also test `1 × 1`, a single row, a single column, and an irregular rectangular image.

## 5. Record results

- [ ] Save both CUDA sources, build and run commands, and a table of correctness results.
- [ ] For each kernel, measure warmed-up kernel time using CUDA events for a few rectangular sizes, including one large enough to launch many blocks. Record the shape, block size, grid size, and kernel time separately from the copies.
- [ ] Explain how to derive a 2D thread mapping from a rectangular shape, why the row stride is the width, and how the blur keeps the output bounds and neighbor bounds separate.
