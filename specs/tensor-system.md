# Tensor System Specification

## Overview

The tensor library (`internal/tensor/`) provides the foundational data structures and operations for the custom inference engine. It implements a pure Go tensor computation framework optimized for CPU inference with SIMD acceleration.

## Design Goals

1. **Pure Go**: No CGO dependencies for maximum portability
2. **Performance**: SIMD-optimized operations (AVX2, NEON)
3. **Memory Efficient**: Support for mmap and quantized data types
4. **Type Safe**: Strong typing for shapes and dimensions
5. **Testable**: 95%+ test coverage with numerical validation

## Core Data Structures

### Tensor

```go
type Tensor struct {
    data   interface{}        // []float32, []float16, []uint8 (quantized)
    shape  []int              // Dimensions [batch, seq_len, hidden_dim, ...]
    stride []int              // Memory layout strides for indexing
    dtype  DataType           // Data type (float32, q4_k, etc.)
    device Device             // CPU (GPU future)
    offset int                // Offset in data array (for views)
}

// DataType represents tensor element type
type DataType int
const (
    Float32 DataType = iota   // 32-bit float (default)
    Float16                   // 16-bit float
    Q4_K                      // 4-bit k-quant
    Q5_K                      // 5-bit k-quant
    Q8_0                      // 8-bit quantization
)

// Device represents compute device
type Device int
const (
    CPU Device = iota
    GPU  // Future: GPU support
)
```

### Constructor Functions

```go
// NewTensor creates a tensor with given shape
func NewTensor(shape []int, dtype DataType) *Tensor

// NewTensorFromData wraps existing data
func NewTensorFromData(data interface{}, shape []int) *Tensor

// NewTensorMmap creates memory-mapped tensor
func NewTensorMmap(path string, offset int64, shape []int, dtype DataType) (*Tensor, error)

// Zeros creates zero-initialized tensor
func Zeros(shape []int, dtype DataType) *Tensor

// Ones creates one-initialized tensor
func Ones(shape []int, dtype DataType) *Tensor
```

## Operations

### Element-wise Operations

```go
// Add performs element-wise addition: C = A + B
func Add(a, b *Tensor) *Tensor

// Mul performs element-wise multiplication: C = A * B
func Mul(a, b *Tensor) *Tensor

// Sub performs element-wise subtraction: C = A - B
func Sub(a, b *Tensor) *Tensor

// Div performs element-wise division: C = A / B
func Div(a, b *Tensor) *Tensor

// Scalar operations
func AddScalar(a *Tensor, scalar float32) *Tensor
func MulScalar(a *Tensor, scalar float32) *Tensor
```

### Reduction Operations

```go
// Sum reduces tensor along dimension(s)
func Sum(a *Tensor, dim int) *Tensor

// Mean computes average along dimension(s)
func Mean(a *Tensor, dim int) *Tensor

// Max finds maximum along dimension(s)
func Max(a *Tensor, dim int) *Tensor

// Min finds minimum along dimension(s)
func Min(a *Tensor, dim int) *Tensor
```

### Shape Manipulation

```go
// Reshape changes tensor shape (no data copy if possible)
func Reshape(a *Tensor, newShape []int) *Tensor

// Transpose swaps two dimensions
func Transpose(a *Tensor, dim0, dim1 int) *Tensor

// Permute reorders all dimensions
func Permute(a *Tensor, dims []int) *Tensor

// Slice extracts subtensor (view, not copy)
func Slice(a *Tensor, dim int, start, end int) *Tensor

// Concat concatenates tensors along dimension
func Concat(tensors []*Tensor, dim int) *Tensor

// Split splits tensor into chunks
func Split(a *Tensor, numChunks int, dim int) []*Tensor
```

### Matrix Operations (Critical for Performance)

```go
// MatMul performs matrix multiplication: C = A @ B
// A: [M x K], B: [K x N] -> C: [M x N]
// Automatically dispatches to fused implementations for quantized tensors
func MatMul(a, b *Tensor) *Tensor

// BatchMatMul for batched matrix multiplication
// A: [B x M x K], B: [B x K x N] -> C: [B x M x N]
func BatchMatMul(a, b *Tensor) *Tensor

// MatVec performs matrix-vector multiplication (optimized)
// A: [M x K], b: [K] -> c: [M]
func MatVec(a, b *Tensor) *Tensor
```

### Fused Quantized Matrix Operations (Phase 10.8+)

**Motivation**: Dequantizing entire weight tensors before MatMul allocates 27.9GB for a 32-layer model. Fused operations eliminate intermediate allocations.

```go
// MatMulQ5K performs fused dequantization + matrix multiplication for Q5_K tensors
// Eliminates the need to create a full intermediate Float32 tensor
//
// Memory savings: 56-69% reduction in allocations
// Quality: Identical to DequantizeQ5_KTensor() + MatMul() (<1e-4 error)
func MatMulQ5K(a *Tensor, bQuantized *Tensor) (*Tensor, error)

// MatMulQ6K performs fused dequantization + matrix multiplication for Q6_K tensors
func MatMulQ6K(a *Tensor, bQuantized *Tensor) (*Tensor, error)
```

**Implementation Status**:
- ✅ Phase 1 (Reference): Correctness-focused naive implementation
- ⏳ Phase 2 (Optimized): Block-wise + SIMD + parallel (target: 2-3x faster)

**Performance Characteristics**:

| Approach | Memory | Speed (256×256) | Quality |
|----------|--------|-----------------|---------|
| Current | 838KB | 2.4ms | Baseline |
| Fused (Phase 1) | 262KB (-69%) | 492ms | Identical |
| Fused (Phase 2 target) | <300KB | <1ms (3x faster) | Identical |

See [FUSED_DEQUANT_MATMUL_PLAN.md](../FUSED_DEQUANT_MATMUL_PLAN.md) and [PHASE1_RESULTS.md](../PHASE1_RESULTS.md) for details.

### Activation Functions

```go
// ReLU: max(0, x)
func ReLU(a *Tensor) *Tensor

// GELU: x * Φ(x) where Φ is Gaussian CDF
func GELU(a *Tensor) *Tensor

// SiLU (Swish): x * sigmoid(x)
func SiLU(a *Tensor) *Tensor

// Softmax: exp(x) / sum(exp(x))
func Softmax(a *Tensor, dim int) *Tensor

// Sigmoid: 1 / (1 + exp(-x))
func Sigmoid(a *Tensor) *Tensor
```

### Normalization

```go
// LayerNorm: (x - mean) / sqrt(variance + eps) * gamma + beta
func LayerNorm(a *Tensor, eps float32) *Tensor

// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
func RMSNorm(a *Tensor, weight *Tensor, eps float32) *Tensor
```

## Matrix Multiplication Implementation

### Naive Implementation (Baseline)

```go
func matmulNaive(a, b *Tensor) *Tensor {
    M, K := a.shape[0], a.shape[1]
    N := b.shape[1]

    result := NewTensor([]int{M, N}, Float32)
    aData := a.data.([]float32)
    bData := b.data.([]float32)
    cData := result.data.([]float32)

    for i := 0; i < M; i++ {
        for j := 0; j < N; j++ {
            sum := float32(0)
            for k := 0; k < K; k++ {
                sum += aData[i*K + k] * bData[k*N + j]
            }
            cData[i*N + j] = sum
        }
    }
    return result
}
```

### Blocked/Tiled Implementation (Cache-Friendly)

```go
func matmulBlocked(a, b *Tensor) *Tensor {
    const blockSize = 32  // Tune for L1 cache

    M, K := a.shape[0], a.shape[1]
    N := b.shape[1]
    result := Zeros([]int{M, N}, Float32)

    // Block outer loops for cache locality
    for i0 := 0; i0 < M; i0 += blockSize {
        for j0 := 0; j0 < N; j0 += blockSize {
            for k0 := 0; k0 < K; k0 += blockSize {
                // Inner micro-kernel
                iMax := min(i0+blockSize, M)
                jMax := min(j0+blockSize, N)
                kMax := min(k0+blockSize, K)

                for i := i0; i < iMax; i++ {
                    for j := j0; j < jMax; j++ {
                        sum := float32(0)
                        for k := k0; k < kMax; k++ {
                            sum += a.At(i, k) * b.At(k, j)
                        }
                        result.Set(i, j, result.At(i, j) + sum)
                    }
                }
            }
        }
    }
    return result
}
```

### Parallel Implementation

```go
func matmulParallel(a, b *Tensor) *Tensor {
    M, K := a.shape[0], a.shape[1]
    N := b.shape[1]
    result := Zeros([]int{M, N}, Float32)

    // Divide work across goroutines
    numWorkers := runtime.NumCPU()
    rowsPerWorker := (M + numWorkers - 1) / numWorkers

    var wg sync.WaitGroup
    for worker := 0; worker < numWorkers; worker++ {
        wg.Add(1)
        startRow := worker * rowsPerWorker
        endRow := min(startRow + rowsPerWorker, M)

        go func(start, end int) {
            defer wg.Done()
            // Compute rows [start, end)
            for i := start; i < end; i++ {
                for j := 0; j < N; j++ {
                    sum := float32(0)
                    for k := 0; k < K; k++ {
                        sum += a.At(i, k) * b.At(k, j)
                    }
                    result.Set(i, j, sum)
                }
            }
        }(startRow, endRow)
    }
    wg.Wait()
    return result
}
```

## SIMD Optimization

### AVX2 (x86-64)

```go
// +build amd64

package tensor

import "github.com/intel-go/cpuid"

var matmulFunc func(*Tensor, *Tensor) *Tensor

func init() {
    // Runtime CPU feature detection
    if cpuid.HasFeature(cpuid.AVX2) {
        matmulFunc = matmulAVX2
    } else if cpuid.HasFeature(cpuid.AVX) {
        matmulFunc = matmulAVX
    } else {
        matmulFunc = matmulNaive
    }
}

// MatMul dispatches to best available implementation
func MatMul(a, b *Tensor) *Tensor {
    return matmulFunc(a, b)
}

// matmulAVX2 uses 256-bit vector instructions
// Implementation can use:
// 1. Pure Go with compiler hints (auto-vectorization)
// 2. Assembly (.s files)
// 3. Unsafe pointer manipulation with careful alignment
func matmulAVX2(a, b *Tensor) *Tensor {
    // Ensure 32-byte alignment for AVX2
    // Use _mm256_* intrinsics conceptually
    // Process 8 float32 values at once

    // For now, rely on compiler auto-vectorization
    // with carefully written loops
    return matmulBlocked(a, b)
}
```

### NEON (ARM64)

```go
// +build arm64

package tensor

func init() {
    // ARM64 always has NEON
    matmulFunc = matmulNEON
}

// matmulNEON uses 128-bit NEON vector instructions
// Process 4 float32 values at once
func matmulNEON(a, b *Tensor) *Tensor {
    // Use NEON intrinsics conceptually
    // or rely on compiler auto-vectorization
    return matmulBlocked(a, b)
}
```

## Quantization

### Q4_K Format

**Status**: ✅ Fully implemented and tested

```go
// Q4_K: 4-bit quantization with k-quant strategy
// Block size: 256 values per block
type Q4_K_Block struct {
    scales  [12]float16   // Scale factors (one per sub-block)
    mins    [12]float16   // Minimum values
    qs      [128]uint8    // Quantized values (4-bit packed)
}

// Quantize converts float32 to Q4_K
func QuantizeQ4_K(src []float32) []Q4_K_Block
// Dequantize converts Q4_K to float32
func DequantizeQ4_K(blocks []Q4_K_Block, dst []float32)

// Tests: 3 tests passing
// - TestQuantizeQ4_K_Roundtrip
// - TestDequantizeQ4_KElement
// - TestDequantizeQ4_KTensor
```

### Q5_K Format

**Status**: ✅ Fully implemented and tested (bug fixed in Phase 10.10)

```go
// Q5_K: 5-bit quantization with k-quant strategy
// Block size: 256 values per super-block (8 sub-blocks of 32 values)
// Each sub-block has its own 6-bit scale and min
type Q5_K_Block struct {
    d     float16      // Super-block scale
    dmin  float16      // Super-block min scale
    scales [12]uint8   // 8 6-bit scales packed into 6 bytes
    qh    [32]uint8    // High bits (5th bit of each value)
    qs    [128]uint8   // Low 4 bits of quantized values
}

// Quantize converts float32 to Q5_K
func QuantizeQ5_K(src []float32) []Q5_K_Block
// Dequantize converts Q5_K to float32
func DequantizeQ5_K(blocks []Q5_K_Block, dst []float32)

// Helper for proper 6-bit scale packing (added in Phase 10.10)
func packScalesAndMins(scales, mins []float32) [12]uint8

// Tests: 14 tests passing
// - All roundtrip tests now pass (bug fixed)
// - Element and tensor dequantization tests
// - Integration tests with quantization tolerance
```

**Phase 10.10 Bug Fix**:
- Fixed critical bug in `QuantizeQ5_K()` scale packing
- Implemented missing `packScalesAndMins()` helper function
- Scales and mins are 6-bit values (0-63) packed into 12 bytes
- Bug was causing loop to set same array indices 8 times instead of packing all values
- All Q5_K tests now passing with proper roundtrip validation

### Q6_K Format

**Status**: ✅ Fully implemented and tested

```go
// Q6_K: 6-bit quantization with k-quant strategy
// Higher quality than Q5_K, still compressed
type Q6_K_Block struct {
    ql    [128]uint8   // Lower 4 bits
    qh    [64]uint8    // Upper 2 bits
    scales [16]int8    // Scales (one per 16 values)
    d     float16      // Block scale
}

// Quantize converts float32 to Q6_K
func QuantizeQ6_K(src []float32) []Q6_K_Block
// Dequantize converts Q6_K to float32
func DequantizeQ6_K(blocks []Q6_K_Block, dst []float32)

// Tests: 18 tests passing
// - Comprehensive roundtrip tests
// - Bit packing tests
// - Element and tensor dequantization
// - Integration tests
```

## Memory Management

### Memory-Mapped Tensors

```go
import (
    "os"
    "syscall"
)

// NewTensorMmap creates a memory-mapped tensor
// This is critical for loading large model files efficiently
func NewTensorMmap(path string, offset int64, size int64, shape []int, dtype DataType) (*Tensor, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }

    // Memory-map the file region
    data, err := syscall.Mmap(
        int(f.Fd()),
        offset,
        int(size),
        syscall.PROT_READ,
        syscall.MAP_SHARED,
    )
    if err != nil {
        f.Close()
        return nil, err
    }

    // Wrap in tensor based on dtype
    var tensorData interface{}
    switch dtype {
    case Float32:
        tensorData = (*[1 << 30]float32)(unsafe.Pointer(&data[0]))[:len(data)/4]
    case Q4_K:
        tensorData = (*[1 << 30]Q4_K_Block)(unsafe.Pointer(&data[0]))[:len(data)/sizeof(Q4_K_Block)]
    // ... other types
    }

    return &Tensor{
        data:   tensorData,
        shape:  shape,
        stride: computeStrides(shape),
        dtype:  dtype,
        device: CPU,
    }, nil
}

// Close unmaps the tensor and releases resources
func (t *Tensor) Close() error {
    if t.mmapFile != nil {
        syscall.Munmap(t.mmapData)
        t.mmapFile.Close()
    }
    return nil
}
```

## Testing Strategy

### Numerical Validation

```go
func TestMatMulAccuracy(t *testing.T) {
    // Test against known results
    a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
    b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

    c := MatMul(a, b)

    expected := []float32{19, 22, 43, 50}  // Hand-computed

    for i := 0; i < 4; i++ {
        if math.Abs(c.data.([]float32)[i] - expected[i]) > 1e-5 {
            t.Errorf("MatMul incorrect at index %d: got %f, want %f",
                i, c.data.([]float32)[i], expected[i])
        }
    }
}

func TestMatMulVsReference(t *testing.T) {
    // Compare against NumPy/PyTorch results
    // Load test data from file generated by Python script
    a, b, expected := loadTestData("testdata/matmul_test_001.npz")

    c := MatMul(a, b)

    // Check all values within tolerance
    for i := range expected {
        diff := math.Abs(c.At(i) - expected[i])
        if diff > 1e-4 {
            t.Errorf("Numerical error at index %d: diff=%e", i, diff)
        }
    }
}
```

### Performance Benchmarks

```go
func BenchmarkMatMul_Small(b *testing.B) {
    a := NewTensor([]int{64, 64}, Float32)
    x := NewTensor([]int{64, 64}, Float32)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        MatMul(a, x)
    }
}

func BenchmarkMatMul_Large(b *testing.B) {
    a := NewTensor([]int{1024, 1024}, Float32)
    x := NewTensor([]int{1024, 1024}, Float32)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        MatMul(a, x)
    }
}

func BenchmarkMatMul_Transformer(b *testing.B) {
    // Typical transformer shapes
    // Q/K/V projections: [seq_len, hidden_dim] @ [hidden_dim, hidden_dim]
    seq := NewTensor([]int{512, 3584}, Float32)  // Qwen 7B hidden_dim
    weight := NewTensor([]int{3584, 3584}, Float32)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        MatMul(seq, weight)
    }
}
```

## Performance Targets

| Operation | Size | Target Performance |
|-----------|------|-------------------|
| MatMul (naive) | 1024x1024 | ~50 ms |
| MatMul (blocked) | 1024x1024 | ~20 ms |
| MatMul (SIMD) | 1024x1024 | ~10 ms |
| MatMul (BLAS baseline) | 1024x1024 | ~5 ms |
| Element-wise | 1M elements | <1 ms |
| Quantize Q4_K | 1M elements | ~5 ms |
| Dequantize Q4_K | 1M elements | ~5 ms |

**Goal**: Achieve 20-50% of optimized BLAS performance for matrix operations.

## Dependencies

- **Standard Library Only**: No external dependencies for core tensor operations
- **Optional**: Assembly for SIMD (can use pure Go initially)
- **Testing**: NumPy/PyTorch for generating reference data

## Fused Dequant-Transpose

The `internal/tensor/dequant_transpose.go` file provides fused functions that
dequantize quantized tensors directly into transposed layout, eliminating the
intermediate full-size allocation that the separate dequant+transpose path requires.

### Functions

```go
func DequantTransposeQ4K(t *Tensor) (*Tensor, error)
func DequantTransposeQ5K(t *Tensor) (*Tensor, error)
func DequantTransposeQ6K(t *Tensor) (*Tensor, error)
```

### Algorithm

1. Allocate a single `[N, M]` float32 result tensor
2. For each quantization block (256 elements):
   a. Parse the block from raw bytes
   b. Dequantize into a small local buffer (1KB, stack-friendly)
   c. Scatter-write the 256 values to their transposed positions
3. Mark result as transposed

### Memory Savings

| Tensor Size | Separate Path | Fused Path | Savings |
|-------------|---------------|------------|---------|
| 3072×3072   | 2 × 36 MB     | 1 × 36 MB | 50%     |
| 3072×8192   | 2 × 96 MB     | 1 × 96 MB | 50%     |

### Integration

`GetOrDequantTranspose()` automatically uses the fused path for 2D Q4_K/Q5_K/Q6_K
tensors and falls back to separate dequant+transpose for other cases.

## Future Optimizations

1. **Assembly micro-kernels** for hot paths (matmul inner loops)
2. **Thread pool** for better goroutine management
3. **Memory allocator** to reduce GC pressure
4. **Kernel fusion** (combine multiple ops into single kernel)
5. **Mixed precision** (FP16 for intermediate computations)
6. **GPU support** via Vulkan/OpenCL (future)

## References

- [BLIS: BLAS-like Library Instantiation Software](https://github.com/flame/blis)
- [Intel MKL Developer Reference](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c)
- [GGML Tensor Library](https://github.com/ggml-org/ggml)
- [PyTorch ATen Tensor Library](https://pytorch.org/cppdocs/)
