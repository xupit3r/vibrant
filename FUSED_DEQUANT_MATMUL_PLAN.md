# Fused Dequantize + MatMul Implementation Plan

**Objective**: Eliminate 8.6GB of memory allocations and achieve 2-3x speedup on quantized weight operations without sacrificing inference quality.

**Status**: Planning Complete - Ready for Implementation
**Priority**: CRITICAL
**Estimated Timeline**: 5-7 days (no rush, methodical approach)

---

## 🎯 Success Criteria

### Performance Targets
- ✅ **Memory**: Reduce allocations from 8.6GB → <500MB per inference pass
- ✅ **Speed**: 2-3x faster than current dequant→matmul approach
- ✅ **Quality**: Numerical accuracy within 1e-4 (bit-exact where possible)

### Validation Requirements
- ✅ All existing tests pass
- ✅ New fused kernel passes reference comparison tests
- ✅ Benchmark shows measurable improvement
- ✅ End-to-end inference produces identical outputs

---

## 📐 Technical Approach

### Current Implementation (Baseline)

```go
// internal/tensor/matmul.go:28-45
func MatMul(a, b *Tensor) *Tensor {
    // Step 1: Check if B is quantized
    if b.DType() == Q5_K {
        // Dequantize ENTIRE tensor to Float32
        bDequant, err := DequantizeQ5_KTensor(b)
        if err != nil {
            panic(err)
        }
        b = bDequant  // Now a full Float32 tensor (huge allocation!)
    }

    // Step 2: Perform MatMul with Float32 tensors
    return matmulSIMDParallel(a, b)
}
```

**Cost Analysis** (for 14B model, 32 layers):
- Attention Q projection: 4096×4096 = 67MB Float32 per layer
- Attention K projection: 4096×4096 = 67MB Float32 per layer
- Attention V projection: 4096×4096 = 67MB Float32 per layer
- Attention output: 4096×4096 = 67MB Float32 per layer
- FFN gate: 4096×12288 = 201MB Float32 per layer
- FFN up: 4096×12288 = 201MB Float32 per layer
- FFN down: 12288×4096 = 201MB Float32 per layer

**Total per layer**: ~872MB Float32 allocations
**Total for 32 layers**: ~27.9GB allocations (!!!)

Note: My earlier estimate of 8.6GB was conservative - actual is worse!

---

### Proposed Fused Implementation

**Key Insight**: We don't need the full dequantized tensor - we only need values as we use them.

```go
// New function: Fused dequantization + matrix multiplication
func MatMulQ5K(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
    // a: [M, K] Float32 (input activations)
    // bQuantized: [K, N] Q5_K (quantized weights)
    // output: [M, N] Float32

    M := a.shape[0]
    K := a.shape[1]
    N := bQuantized.shape[1]

    output := NewTensor([]int{M, N}, Float32)

    // For each output element
    for i := 0; i < M; i++ {
        for j := 0; j < N; j++ {
            sum := float32(0.0)

            // Dot product: a[i,:] · b[:,j]
            for k := 0; k < K; k++ {
                aVal := a.At(i, k)

                // Dequantize single element on-demand
                bVal := DequantizeQ5_KElement(bQuantized.data.([]byte), k*N + j)

                sum += aVal * bVal
            }

            output.Set(sum, i, j)
        }
    }

    return output, nil
}
```

**Benefits**:
- ✅ Zero intermediate Float32 tensor allocation
- ✅ Better cache locality (access quantized data sequentially)
- ✅ Exact same numerical result (same dequant math)

**Optimizations** (to be added):
1. **Block-wise processing**: Process B in blocks for better cache usage
2. **SIMD vectorization**: Vectorize the inner loop
3. **Parallelization**: Parallelize outer loops
4. **Prefetching**: Prefetch quantized blocks

---

## 🏗️ Implementation Plan

### Phase 1: Foundation (Days 1-2)

#### Task 1.1: Create Reference Implementation
**File**: `internal/tensor/matmul_quant.go` (new file)

```go
package tensor

// MatMulQ5K performs fused dequantization + matrix multiplication for Q5_K tensors.
// This is the reference implementation (correctness over speed).
func MatMulQ5K(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
    // Validate inputs
    // Implement naive triple-loop version
    // Focus on correctness
}

// MatMulQ6K performs fused dequantization + matrix multiplication for Q6_K tensors.
func MatMulQ6K(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
    // Similar to Q5_K
}
```

**Deliverable**: Working naive implementation

#### Task 1.2: Create Test Suite
**File**: `internal/tensor/matmul_quant_test.go` (new file)

**Tests to implement**:

1. **Correctness Tests** (vs reference):
```go
func TestMatMulQ5K_Correctness(t *testing.T) {
    // Create small test matrix
    a := NewTensor([]int{4, 8}, Float32)
    // Fill with known values

    // Create Q5_K quantized matrix
    bFloat := NewTensor([]int{8, 4}, Float32)
    // Fill with known values
    bQuant := QuantizeQ5_K(bFloat)

    // Method 1: Current approach (dequant then matmul)
    bDequant := DequantizeQ5_KTensor(bQuant)
    expected := MatMul(a, bDequant)

    // Method 2: Fused approach
    result := MatMulQ5K(a, bQuant)

    // Compare results
    maxDiff := MaxAbsDiff(expected, result)
    if maxDiff > 1e-4 {
        t.Errorf("Max difference %f exceeds threshold 1e-4", maxDiff)
    }
}
```

2. **Size Validation Tests**:
```go
func TestMatMulQ5K_SizeMismatch(t *testing.T)
func TestMatMulQ5K_ZeroMatrix(t *testing.T)
func TestMatMulQ5K_Identity(t *testing.T)
```

3. **Edge Case Tests**:
```go
func TestMatMulQ5K_SmallMatrix(t *testing.T)    // 2x2
func TestMatMulQ5K_LargeMatrix(t *testing.T)    // 1024x1024
func TestMatMulQ5K_NonSquare(t *testing.T)      // 256x512
```

4. **Performance Benchmarks**:
```go
func BenchmarkMatMulQ5K_vs_Current(b *testing.B) {
    // Compare: Fused vs (Dequant + MatMul)
    // Measure: Time and allocations
}

func BenchmarkMatMulQ5K_MemoryProfile(b *testing.B) {
    // Track allocations
}
```

**Deliverable**: Comprehensive test suite

---

### Phase 2: Optimization (Days 3-4)

#### Task 2.1: Block-wise Processing

**Optimize memory access patterns**:

```go
func MatMulQ5KBlocked(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
    const BLOCK_SIZE = 256  // Q5_K block size

    // Process B in blocks for better cache locality
    for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
        // Dequantize one block at a time
        block := dequantizeBlock(bQuantized, blockIdx)

        // Use dequantized block for multiple output elements
        // before moving to next block
    }
}
```

**Expected**: 1.5-2x speedup from better cache usage

#### Task 2.2: SIMD Vectorization

**Vectorize inner loop** (platform-specific):

```go
// matmul_quant_simd_amd64.go
func matmulQ5KInnerLoopSIMD(a, b []float32, result *float32, n int) {
    // Use AVX2 for vectorized multiply-accumulate
    // Process 8 elements at a time
}
```

**Expected**: 2-3x speedup on inner loop

#### Task 2.3: Parallelization

**Parallelize outer loop**:

```go
func MatMulQ5KParallel(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
    // Parallelize over output rows
    var wg sync.WaitGroup
    numWorkers := runtime.NumCPU()

    for workerID := 0; workerID < numWorkers; workerID++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // Process rows [start, end)
        }(workerID)
    }

    wg.Wait()
    return output, nil
}
```

**Expected**: 3-4x speedup on multi-core

**Cumulative expected speedup**: 2-3x over current approach

---

### Phase 3: Integration (Days 5-6)

#### Task 3.1: Update MatMul Dispatcher

**File**: `internal/tensor/matmul.go`

```go
func MatMul(a, b *Tensor) *Tensor {
    // Dispatch to appropriate implementation based on types

    // Fast path: Both Float32 (existing code)
    if a.DType() == Float32 && b.DType() == Float32 {
        return matmulSIMDParallel(a, b)
    }

    // Fused path: A is Float32, B is quantized
    if a.DType() == Float32 {
        switch b.DType() {
        case Q5_K:
            result, err := MatMulQ5KOptimized(a, b)
            if err != nil {
                panic(err)
            }
            return result
        case Q6_K:
            result, err := MatMulQ6KOptimized(a, b)
            if err != nil {
                panic(err)
            }
            return result
        }
    }

    // Fallback: Dequantize and use Float32 path (for other quant types)
    if b.DType() != Float32 {
        bDequant, err := DequantizeTensor(b)
        if err != nil {
            panic(err)
        }
        b = bDequant
    }

    return matmulSIMDParallel(a, b)
}
```

#### Task 3.2: Update Tensor.At() Support

Ensure Q5_K/Q6_K tensors work seamlessly:

```go
// internal/tensor/tensor.go
func (t *Tensor) At(indices ...int) float32 {
    switch t.dtype {
    case Float32:
        return t.data.([]float32)[idx]
    case Float16:
        return float16ToFloat32(t.data.([]uint16)[idx])
    case Q5_K:
        return DequantizeQ5_KElement(t.data.([]byte), idx)  // Already implemented
    case Q6_K:
        return DequantizeQ6_KElement(t.data.([]byte), idx)  // Already implemented
    default:
        panic(fmt.Sprintf("At() not supported for dtype %s", t.dtype))
    }
}
```

---

### Phase 4: Validation & Benchmarking (Day 7)

#### Task 4.1: Integration Tests

**File**: `internal/transformer/layer_test.go` (update existing)

```go
func TestTransformerLayerWithQ5K(t *testing.T) {
    // Load real Q5_K weights from GGUF
    // Run forward pass
    // Compare with reference (dequant + matmul)
    // Verify max difference < 1e-4
}
```

#### Task 4.2: End-to-End Validation

**File**: `test/integration/inference_quality_test.go` (new)

```go
func TestInferenceQuality_FusedVsReference(t *testing.T) {
    // Generate 100 tokens with both implementations
    // Compare token-by-token
    // Should be identical (deterministic with same seed)
}
```

#### Task 4.3: Performance Benchmarks

**Run comprehensive benchmarks**:

```bash
# Baseline (current approach)
go test ./internal/tensor -bench=MatMul -benchmem > baseline.txt

# Fused implementation
go test ./internal/tensor -bench=MatMulQ5K -benchmem > fused.txt

# Compare
benchstat baseline.txt fused.txt
```

**Expected output**:
```
name              old time/op    new time/op    delta
MatMul_256x256    3.51ms ± 2%    1.20ms ± 3%   -65.81%  (p=0.000)

name              old alloc/op   new alloc/op   delta
MatMul_256x256    268MB ± 0%     0.5MB ± 0%    -99.81%  (p=0.000)

name              old allocs/op  new allocs/op  delta
MatMul_256x256    39.0 ± 0%      5.0 ± 0%      -87.18%  (p=0.000)
```

#### Task 4.4: Memory Profiling

```bash
# Profile memory allocations
go test ./internal/transformer -run TestForwardPass -memprofile=mem.prof
go tool pprof -http=:8080 mem.prof

# Verify: No large Float32 tensor allocations from dequant
```

---

## 📊 Testing Strategy

### Level 1: Unit Tests (Fast, Isolated)
- ✅ Correctness vs reference implementation
- ✅ Edge cases (zero, identity, size mismatches)
- ✅ All quantization types (Q5_K, Q6_K)
- **Run**: Every code change
- **Coverage target**: 95%+

### Level 2: Integration Tests (Medium, Realistic)
- ✅ Full transformer layer forward pass
- ✅ Attention mechanism with Q5_K weights
- ✅ FFN with Q5_K weights
- **Run**: Before commit
- **Coverage target**: Key code paths

### Level 3: End-to-End Tests (Slow, Complete)
- ✅ Generate 100+ tokens
- ✅ Compare outputs with reference
- ✅ Verify deterministic results
- **Run**: Before merge/release
- **Coverage target**: Real-world scenarios

### Level 4: Performance Tests (Benchmarks)
- ✅ Speed comparison (fused vs current)
- ✅ Memory comparison (allocations)
- ✅ Scaling tests (small to large matrices)
- **Run**: After optimization changes
- **Coverage target**: Performance regressions caught

### Level 5: Quality Validation (Critical)
- ✅ Numerical accuracy < 1e-4
- ✅ Output comparison with llama.cpp (if possible)
- ✅ Long sequence generation (1000+ tokens)
- **Run**: Final validation
- **Coverage target**: Production-ready quality

---

## 🎯 Implementation Checklist

### Phase 1: Foundation ☐
- ☐ Create `internal/tensor/matmul_quant.go`
- ☐ Implement `MatMulQ5K()` reference version
- ☐ Implement `MatMulQ6K()` reference version
- ☐ Create `internal/tensor/matmul_quant_test.go`
- ☐ Write correctness tests (vs current implementation)
- ☐ Write edge case tests
- ☐ Write initial benchmarks
- ☐ All tests passing ✅

### Phase 2: Optimization ☐
- ☐ Implement block-wise processing
- ☐ Benchmark block-wise vs naive
- ☐ Implement SIMD inner loop (AVX2)
- ☐ Benchmark SIMD vs scalar
- ☐ Implement parallelization
- ☐ Benchmark parallel vs serial
- ☐ All tests still passing ✅

### Phase 3: Integration ☐
- ☐ Update `MatMul()` dispatcher in matmul.go
- ☐ Add feature flag for gradual rollout
- ☐ Update transformer tests
- ☐ All existing tests passing ✅

### Phase 4: Validation ☐
- ☐ Run full test suite
- ☐ Memory profiling (no large allocations)
- ☐ Performance benchmarks (2-3x speedup confirmed)
- ☐ End-to-end quality tests (accuracy validated)
- ☐ Document results in summary

---

## 🚀 Expected Results

### Performance Improvements

**Memory** (per 32-layer inference):
- Current: ~27.9GB Float32 allocations
- Target: <500MB allocations
- **Improvement**: ~98% reduction 🎉

**Speed** (256x256 MatMul, typical attention size):
- Current: ~3.5ms (matmul) + dequant overhead
- Target: ~1.2ms (fused)
- **Improvement**: ~2-3x faster 🚀

**Allocations** (per MatMul operation):
- Current: 268MB + metadata allocations
- Target: <1MB
- **Improvement**: ~99% reduction 🎉

### Quality Preservation

- ✅ Numerical accuracy: < 1e-4 max difference
- ✅ Deterministic: Same outputs with same seed
- ✅ Bit-exact: Where mathematically equivalent

---

## 📝 Documentation Updates

After implementation, update:

1. **PHASE10.8_SUMMARY.md**: Add fused matmul achievement
2. **specs/tensor-system.md**: Document fused operations
3. **PERFORMANCE_OPTIMIZATION_PLAN.md**: Mark as complete
4. **README.md**: Highlight memory efficiency improvements

---

## 🤔 Open Questions

1. **Block size tuning**: Should we auto-tune block size based on CPU cache?
2. **ARM support**: Implement NEON version in Phase 2 or defer?
3. **Q4_K/Q8_0**: Extend fused approach to other quant types immediately?
4. **Feature flag**: Use environment variable for gradual rollout?

---

**Ready to start implementation!**
