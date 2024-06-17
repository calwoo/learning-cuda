#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rgb_to_grayscale_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int inputOffset = (row * width + col) * channels;
        int outputOffset = row * width + col;

        unsigned char r = input[inputOffset];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];

        output[outputOffset] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
    }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
