// Vibrant CUDA Kernels
// Basic operations for tensor computation on NVIDIA GPUs

#ifndef VIBRANT_CUDA_KERNELS_H
#define VIBRANT_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Thread block dimensions
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", \
                    __FUNCTION__, __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
extern "C" __global__ void matmul_f32(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

// Optimized matrix multiplication for single row (decode phase)
// A: [1 x K], B: [K x N], C: [1 x N]
extern "C" __global__ void matmul_f32_single_row(
    const float* A,
    const float* B,
    float* C,
    int N,
    int K
);

// ============================================================================
// Normalization Operations
// ============================================================================

// Softmax: output[i] = exp(input[i]) / sum(exp(input))
// Numerically stable implementation (subtract max)
extern "C" __global__ void softmax_f32(
    const float* input,
    float* output,
    int size
);

// Batched softmax for attention
// input/output: [batch_size x size]
extern "C" __global__ void softmax_batched_f32(
    const float* input,
    float* output,
    int batch_size,
    int size
);

// RMS normalization: output = input / sqrt(mean(input^2) + eps) * weight
extern "C" __global__ void rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    int size,
    float eps
);

// Batched RMS normalization
extern "C" __global__ void rms_norm_batched_f32(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int size,
    float eps
);

// ============================================================================
// Element-wise Operations
// ============================================================================

// Element-wise addition: C = A + B
extern "C" __global__ void add_f32(
    const float* A,
    const float* B,
    float* C,
    int size
);

// Element-wise multiplication: C = A * B
extern "C" __global__ void mul_f32(
    const float* A,
    const float* B,
    float* C,
    int size
);

// Scalar multiplication: B = A * scalar
extern "C" __global__ void mul_scalar_f32(
    const float* A,
    float scalar,
    float* B,
    int size
);

// SiLU activation: output = input * sigmoid(input)
// Also known as Swish
extern "C" __global__ void silu_f32(
    const float* input,
    float* output,
    int size
);

// Copy tensor data
extern "C" __global__ void copy_f32(
    const float* src,
    float* dst,
    int size
);

#endif // VIBRANT_CUDA_KERNELS_H
