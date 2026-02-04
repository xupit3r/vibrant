#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A @ B
// A: [M x K], B: [K x N] -> C: [M x N]
// Uses tiled algorithm with threadgroup memory for better performance
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Tile size (must match dispatch parameters)
    const uint TILE_SIZE = 16;
    
    // Shared memory for tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0f;
    
    // Number of tiles needed
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        uint aRow = tgid.y * TILE_SIZE + tid.y;
        uint aCol = t * TILE_SIZE + tid.x;
        if (aRow < M && aCol < K) {
            As[tid.y][tid.x] = A[aRow * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        uint bRow = t * TILE_SIZE + tid.y;
        uint bCol = tgid.x * TILE_SIZE + tid.x;
        if (bRow < K && bCol < N) {
            Bs[tid.y][tid.x] = B[bRow * N + bCol];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized single-row matrix multiplication for decode step (M=1)
// C[1 x N] = A[1 x K] @ B[K x N]
kernel void matmul_f32_single_row(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[k] * B[k * N + tid];
    }
    C[tid] = sum;
}

// Softmax: output[i] = exp(input[i]) / sum(exp(input))
// Applied along the last dimension
// Two-pass algorithm: first find max, then compute exp and normalize
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    
    // First pass: find max (for numerical stability)
    float maxVal = input[0];
    for (uint i = 1; i < size; i++) {
        maxVal = max(maxVal, input[i]);
    }
    
    // Second pass: compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += exp(input[i] - maxVal);
    }
    
    // Normalize
    output[tid] = exp(input[tid] - maxVal) / sum;
}

// Optimized softmax for batched sequences
// Input/output shape: [batch_size x seq_len]
kernel void softmax_batched_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint pos = gid.x;
    
    if (batch >= batch_size || pos >= seq_len) return;
    
    uint offset = batch * seq_len;
    device const float* row = input + offset;
    device float* out_row = output + offset;
    
    // Find max for this row
    float maxVal = row[0];
    for (uint i = 1; i < seq_len; i++) {
        maxVal = max(maxVal, row[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < seq_len; i++) {
        sum += exp(row[i] - maxVal);
    }
    
    // Normalize this element
    out_row[pos] = exp(row[pos] - maxVal) / sum;
}

// RMS Normalization: output[i] = input[i] / rms * weight[i]
// where rms = sqrt(mean(input^2) + eps)
kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    
    // Compute RMS
    float sum_sq = 0.0f;
    for (uint i = 0; i < size; i++) {
        float val = input[i];
        sum_sq += val * val;
    }
    float rms = sqrt(sum_sq / float(size) + eps);
    
    // Normalize and scale
    output[tid] = (input[tid] / rms) * weight[tid];
}

// Batched RMS normalization
// Input/output shape: [batch_size x dim]
kernel void rms_norm_batched_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint pos = gid.x;
    
    if (batch >= batch_size || pos >= dim) return;
    
    uint offset = batch * dim;
    device const float* row = input + offset;
    device float* out_row = output + offset;
    
    // Compute RMS for this row
    float sum_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = row[i];
        sum_sq += val * val;
    }
    float rms = sqrt(sum_sq / float(dim) + eps);
    
    // Normalize and scale this element
    out_row[pos] = (row[pos] / rms) * weight[pos];
}

// Element-wise addition: C = A + B
kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    C[tid] = A[tid] + B[tid];
}

// Element-wise multiplication: C = A * B
kernel void mul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    C[tid] = A[tid] * B[tid];
}

// Element-wise scalar multiplication: B = A * scalar
kernel void mul_scalar_f32(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    B[tid] = A[tid] * scalar;
}

// SiLU activation (Swish): output = x * sigmoid(x)
kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    float x = input[tid];
    output[tid] = x / (1.0f + exp(-x));
}

// Copy buffer: dst = src
kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= size) return;
    dst[tid] = src[tid];
}
