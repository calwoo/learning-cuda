import triton
import triton.language as tl


@triton.jit
def affine_kernel(x_ptr, b_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    # TODO: Compute this program's offsets and the mask for the tail.
    # TODO: Load x and the per-element bias, then compute alpha * x + b.
    # TODO: Store the result using the same mask.
    pass
