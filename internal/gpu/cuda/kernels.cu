// Vibrant CUDA Kernel Implementations

#include "kernels.h"
#include <stdio.h>

// ============================================================================
// Matrix Operations
// ============================================================================

// General matrix multiplication with shared memory tiling
__global__ void matmul_f32(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Shared memory for tile caching
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized for single row (1 x K) * (K x N) -> (1 x N)
// Critical for decoder phase in transformer inference
__global__ void matmul_f32_single_row(
    const float* A,
    const float* B,
    float* C,
    int N,
    int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B[k * N + col];
        }
        C[col] = sum;
    }
}

// ============================================================================
// Normalization Operations
// ============================================================================

// Numerically stable softmax (two-pass: find max, then exp/normalize)
__global__ void softmax_f32(
    const float* input,
    float* output,
    int size
) {
    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = &shared[1];

    // Find maximum value (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = input[i];
        if (val > thread_max) {
            thread_max = val;
        }
    }

    // Reduce to find global max
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (max_val[threadIdx.x + s] > max_val[threadIdx.x]) {
                max_val[threadIdx.x] = max_val[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float max_value = max_val[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = expf(input[i] - max_value);
        output[i] = val;
        thread_sum += val;
    }

    // Reduce to find total sum
    sum_val[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_val[threadIdx.x] += sum_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = sum_val[0];
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] /= total_sum;
    }
}

// Batched softmax (independent softmax for each batch)
__global__ void softmax_batched_f32(
    const float* input,
    float* output,
    int batch_size,
    int size
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    const float* in_batch = input + batch * size;
    float* out_batch = output + batch * size;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = &shared[blockDim.x];

    // Find maximum
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = in_batch[i];
        if (val > thread_max) thread_max = val;
    }

    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (max_val[threadIdx.x + s] > max_val[threadIdx.x]) {
                max_val[threadIdx.x] = max_val[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float max_value = max_val[0];
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = expf(in_batch[i] - max_value);
        out_batch[i] = val;
        thread_sum += val;
    }

    sum_val[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_val[threadIdx.x] += sum_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = sum_val[0];
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out_batch[i] /= total_sum;
    }
}

// RMS normalization
__global__ void rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    int size,
    float eps
) {
    extern __shared__ float shared_mem[];

    // Compute mean of squares
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = input[i];
        thread_sum += val * val;
    }

    // Reduce sum
    shared_mem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_square = shared_mem[0] / size;
    float rms = sqrtf(mean_square + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = (input[i] / rms) * weight[i];
    }
}

// Batched RMS normalization
__global__ void rms_norm_batched_f32(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int size,
    float eps
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    const float* in_batch = input + batch * size;
    float* out_batch = output + batch * size;

    extern __shared__ float shared_mem[];

    // Compute mean of squares
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = in_batch[i];
        thread_sum += val * val;
    }

    shared_mem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_square = shared_mem[0] / size;
    float rms = sqrtf(mean_square + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out_batch[i] = (in_batch[i] / rms) * weight[i];
    }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

__global__ void add_f32(
    const float* A,
    const float* B,
    float* C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void mul_f32(
    const float* A,
    const float* B,
    float* C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void mul_scalar_f32(
    const float* A,
    float scalar,
    float* B,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        B[idx] = A[idx] * scalar;
    }
}

__global__ void silu_f32(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        output[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void copy_f32(
    const float* src,
    float* dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

// Apply RoPE to input tensor using INTERLEAVED PAIRS
// Input: [batch_size * num_heads * seq_len * head_dim]
// Memory layout: pairs are interleaved (x0, x1, x2, x3...) where (x0,x1), (x2,x3) are pairs
// cosTable/sinTable: precomputed cos/sin values [maxSeqLen * halfDim]
// positions: position indices for each token [seq_len]
__global__ void rope_f32(
    const float* input,
    float* output,
    const float* cosTable,
    const float* sinTable,
    const int* positions,
    int batchSize,
    int numHeads,
    int seqLen,
    int headDim,
    int halfDim
) {
    // Each thread handles ONE PAIR of elements (not single element!)
    // Total pairs = batchSize * numHeads * seqLen * halfDim
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = batchSize * numHeads * seqLen * halfDim;
    
    if (pairIdx < totalPairs) {
        // Decode indices: which batch, head, sequence position, and pair within head_dim
        int i = pairIdx % halfDim;          // Pair index within head_dim (0 to halfDim-1)
        int s = (pairIdx / halfDim) % seqLen;
        int h = (pairIdx / (halfDim * seqLen)) % numHeads;
        int b = pairIdx / (halfDim * seqLen * numHeads);
        
        // Get position for this token
        int pos = positions[s];
        
        // Table lookup
        int tableIdx = pos * halfDim + i;
        float c = cosTable[tableIdx];
        float sn = sinTable[tableIdx];
        
        // Compute base offset for this sequence position in the tensor
        // Layout: [batch][head][seq][dim]
        int baseOffset = b * (numHeads * seqLen * headDim) + 
                        h * (seqLen * headDim) + 
                        s * headDim;
        
        // Get indices for the interleaved pair
        // Pairs are at: (0,1), (2,3), (4,5), ..., (2*i, 2*i+1)
        int idx0 = baseOffset + 2 * i;      // Even index (first of pair)
        int idx1 = baseOffset + 2 * i + 1;  // Odd index (second of pair)
        
        // Read the pair
        float x0 = input[idx0];
        float x1 = input[idx1];
        
        // Apply rotation and write back
        output[idx0] = x0 * c - x1 * sn;
        output[idx1] = x0 * sn + x1 * c;
    }
}
