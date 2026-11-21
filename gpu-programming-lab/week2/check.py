"""Run the Week 2 Triton kernels against PyTorch reference operations."""

import argparse
import os

import torch
import triton

from affine import affine_kernel
from vector_add import vector_add_kernel


BLOCK_SIZES = (128, 256, 512, 1024)


def check(exercise: str, device: str) -> None:
    torch.manual_seed(42)
    alpha = -0.375
    for block_size in BLOCK_SIZES:
        lengths = sorted({1, block_size - 1, block_size, block_size + 1, 1025, 65537})
        for n in lengths:
            x = torch.randn(n, device=device, dtype=torch.float32)
            b = torch.randn_like(x)
            # A missing store stays NaN and fails the comparison.
            out = torch.full_like(x, float("nan"))
            grid = (triton.cdiv(n, block_size),)

            if exercise == "vector_add":
                vector_add_kernel[grid](x, b, out, n, BLOCK_SIZE=block_size)
                expected = x + b
            else:
                affine_kernel[grid](x, b, out, n, alpha, BLOCK_SIZE=block_size)
                expected = alpha * x + b

            try:
                # Allow for rounding differences if the affine kernel uses FMA.
                torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
            except AssertionError as error:
                raise AssertionError(
                    f"{exercise} failed for n={n}, block_size={block_size}"
                ) from error

    print(f"{exercise}: all shape and block-size checks passed on {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=("vector_add", "affine"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    interpreting = os.environ.get("TRITON_INTERPRET") == "1"
    if args.device == "cpu" and not interpreting:
        parser.error("Set TRITON_INTERPRET=1 before running CPU checks.")
    if args.device == "cuda" and (interpreting or not torch.cuda.is_available()):
        parser.error("CUDA checks require a CUDA-enabled PyTorch build and interpreter mode off.")

    check(args.exercise, args.device)


if __name__ == "__main__":
    main()
