import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # TODO: Compute the vector offsets owned by this program instance.
    # TODO: Mask out offsets beyond n when loading the two inputs.
    # TODO: Add the inputs and store the valid elements.
    pass
