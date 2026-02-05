// Vibrant CUDA Kernel Launch Wrappers
// This file contains C++ wrapper functions that handle grid/block dimension
// calculations and launch the CUDA kernels defined in kernels.cu

#include "kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Helper function to calculate optimal grid and block dimensions
static dim3 calculateGrid2D(int M, int N, int blockSize) {
    int gridX = (N + blockSize - 1) / blockSize;
    int gridY = (M + blockSize - 1) / blockSize;
    return dim3(gridX, gridY, 1);
}

static dim3 calculateGrid1D(int N, int blockSize) {
    int gridX = (N + blockSize - 1) / blockSize;
    return dim3(gridX, 1, 1);
}

// ============================================================================
// Matrix Operations
// ============================================================================

extern "C" void matmul_f32_launch(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Use TILE_SIZE x TILE_SIZE thread blocks
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim = calculateGrid2D(M, N, TILE_SIZE);
    
    matmul_f32<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}

extern "C" void matmul_f32_single_row_launch(
    const float* A, const float* B, float* C,
    int N, int K,
    cudaStream_t stream
) {
    // Use 256 threads per block for single-row matmul
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(N, blockSize);
    
    matmul_f32_single_row<<<gridDim, blockDim, 0, stream>>>(A, B, C, N, K);
}

// ============================================================================
// Normalization Operations
// ============================================================================

extern "C" void softmax_f32_launch(
    const float* input, float* output,
    int size,
    cudaStream_t stream
) {
    // Single block with 256 threads, shared memory for reduction
    int blockSize = 256;
    int sharedMemSize = 2 * sizeof(float); // max_val and sum_val
    
    softmax_f32<<<1, blockSize, sharedMemSize, stream>>>(input, output, size);
}

extern "C" void softmax_batched_f32_launch(
    const float* input, float* output,
    int batch_size, int size,
    cudaStream_t stream
) {
    // One block per batch element
    int blockSize = 256;
    int sharedMemSize = 2 * sizeof(float) * blockSize; // Per-thread max and sum
    
    softmax_batched_f32<<<batch_size, blockSize, sharedMemSize, stream>>>(input, output, batch_size, size);
}

extern "C" void rms_norm_f32_launch(
    const float* input, const float* weight, float* output,
    int size, float eps,
    cudaStream_t stream
) {
    // Single block with 256 threads
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float); // For reduction
    
    rms_norm_f32<<<1, blockSize, sharedMemSize, stream>>>(input, weight, output, size, eps);
}

extern "C" void rms_norm_batched_f32_launch(
    const float* input, const float* weight, float* output,
    int batch_size, int size, float eps,
    cudaStream_t stream
) {
    // One block per batch element
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float);
    
    rms_norm_batched_f32<<<batch_size, blockSize, sharedMemSize, stream>>>(input, weight, output, batch_size, size, eps);
}

// ============================================================================
// Element-wise Operations
// ============================================================================

extern "C" void add_f32_launch(
    const float* A, const float* B, float* C,
    int size,
    cudaStream_t stream
) {
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(size, blockSize);
    
    add_f32<<<gridDim, blockDim, 0, stream>>>(A, B, C, size);
}

extern "C" void mul_f32_launch(
    const float* A, const float* B, float* C,
    int size,
    cudaStream_t stream
) {
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(size, blockSize);
    
    mul_f32<<<gridDim, blockDim, 0, stream>>>(A, B, C, size);
}

extern "C" void mul_scalar_f32_launch(
    const float* A, float scalar, float* B,
    int size,
    cudaStream_t stream
) {
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(size, blockSize);
    
    mul_scalar_f32<<<gridDim, blockDim, 0, stream>>>(A, scalar, B, size);
}

extern "C" void silu_f32_launch(
    const float* input, float* output,
    int size,
    cudaStream_t stream
) {
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(size, blockSize);
    
    silu_f32<<<gridDim, blockDim, 0, stream>>>(input, output, size);
}

extern "C" void copy_f32_launch(
    const float* src, float* dst,
    int size,
    cudaStream_t stream
) {
    int blockSize = 256;
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim = calculateGrid1D(size, blockSize);
    
    copy_f32<<<gridDim, blockDim, 0, stream>>>(src, dst, size);
}
