# Phase 10.8: Optimization & Quantization - Roadmap

**Status**: IN PROGRESS 🚧
**Start Date**: January 31, 2026
**Priority**: HIGH - Blocks functional inference

## Executive Summary

Phase 10.8 focuses on implementing the critical missing pieces for functional inference and optimizing performance. The primary blocker is **quantized tensor support**, which must be implemented before any meaningful inference can occur.

## Critical Path (Blocking Inference)

### 1. Quantized Tensor Dequantization ⚠️ **HIGHEST PRIORITY**

**Status**: Not started
**Blocks**: All inference functionality
**Estimated Time**: 2-3 days

#### Task 1.1: Understand Q5_K Format

Study GGUF Q5_K quantization format:
- Block structure (32 floats per block)
- Quantization parameters
- Bit packing scheme
- Scale factors

**Resources**:
- llama.cpp source: `ggml-quants.c`
- GGUF specification
- Example Q5_K tensors

#### Task 1.2: Implement Q5_K Dequantization

```go
// internal/tensor/quant_q5k.go
package tensor

// Q5_K block structure (based on llama.cpp)
type BlockQ5_K struct {
    d     float16    // Delta (scale factor)
    dmin  float16    // Min delta
    scales [12]uint8 // Quantized scales
    qs    [32]uint8 // Quantized values (5-bit packed)
    qh    [4]uint8  // High bits
}

// DequantizeQ5_K converts Q5_K blocks to float32
func DequantizeQ5_K(blocks []BlockQ5_K, output []float32) error {
    // Implementation based on llama.cpp
    // Unpack 5-bit values
    // Apply scales
    // Reconstruct floats
}
```

**Tests**:
- Known value tests (compare with llama.cpp)
- Roundtrip tests (quant → dequant)
- Numerical accuracy (within tolerance)
- Benchmark performance

#### Task 1.3: Implement Other Quantization Types

Priority order:
1. ✅ Q5_K (most common for 14B models)
2. Q4_K (smaller models, faster)
3. Q6_K (higher quality)
4. Q8_0 (baseline quantization)

#### Task 1.4: Integrate Dequantization into Tensor.At()

```go
// internal/tensor/tensor.go
func (t *Tensor) At(indices ...int) float32 {
    switch t.dtype {
    case Float32:
        // Existing implementation
    case Q5_K:
        // Dequantize on-the-fly
        blockIdx := calculateBlockIndex(indices, t.shape)
        offset := calculateOffsetInBlock(indices)
        return dequantizeQ5KElement(t.data, blockIdx, offset)
    // ... other types
    }
}
```

**Options**:
1. **On-the-fly** (simple, slow): Dequantize each access
2. **Block caching** (medium): Cache dequantized blocks
3. **Eager** (fast, memory): Dequantize whole tensor on load

**Recommendation**: Start with eager for embeddings, on-the-fly for weights

#### Task 1.5: Optimize Dequantization

- SIMD vectorization
- Parallel dequantization
- Block-wise processing
- Minimize allocations

**Target**: Dequantize embedding weights (535MB) in <100ms

### 2. Quantized Matrix Multiplication 🔥 **HIGH PRIORITY**

**Status**: Not started
**Depends on**: Task 1 (Dequantization)
**Estimated Time**: 3-4 days

#### Task 2.1: Naive Quantized MatMul

```go
// internal/tensor/matmul_quant.go
func MatMulQ5K(a *Tensor, b *Tensor) (*Tensor, error) {
    // Dequantize B on-the-fly during multiplication
    // Compute A * dequant(B)
    // More memory efficient than full dequant
}
```

#### Task 2.2: Optimized Quantized MatMul

- Fused dequant + multiply
- SIMD for dequantization
- Block-wise computation
- Cache-friendly access patterns

**Target**: Within 2-3x of full Float32 MatMul

### 3. Memory Optimization 💾 **MEDIUM PRIORITY**

**Status**: Not started
**Estimated Time**: 2 days

#### Task 3.1: Tensor Memory Pooling

Reduce GC pressure:
- Pre-allocate tensor pool
- Reuse temp tensors
- Arena allocator for activations

#### Task 3.2: KV-Cache Optimization

- Pre-allocate cache buffers
- Avoid repeated allocations
- Efficient cache updates

### 4. SIMD Optimization 🚀 **MEDIUM PRIORITY**

**Status**: Partial (compiler auto-vectorization)
**Estimated Time**: 3-4 days

#### Task 4.1: Manual SIMD for Hot Paths

Identify hot paths from profiling:
- Dequantization loops
- Element-wise operations
- Dot products
- Reductions (sum, max)

#### Task 4.2: Platform-Specific Assembly

If profiling shows benefit:
- AVX2 for x86_64
- NEON for ARM64
- Benchmark gains vs complexity

## Secondary Optimizations

### 5. Attention Mechanism Optimization

- Flash Attention (later phase)
- Grouped-Query Attention efficiency
- KV-cache preallocation
- Parallel head computation

### 6. Feed-Forward Optimization

- Fused SwiGLU operations
- Efficient tensor reshaping
- Minimize intermediate allocations

### 7. Sampling Optimization

- ✅ Already efficient (< 5ms)
- Possible improvements:
  - SIMD softmax
  - Faster top-k heap
  - Optimized top-p scan

## Implementation Plan

### Week 1: Quantization Foundation

**Days 1-2**: Q5_K dequantization
- Study format
- Implement dequantization
- Write tests
- Benchmark

**Days 3-4**: Integration
- Update Tensor.At()
- Test with real model
- Verify numerical accuracy
- Profile performance

**Day 5**: Other quant types
- Q4_K implementation
- Q6_K implementation
- Q8_0 implementation

### Week 2: Optimization

**Days 1-2**: Quantized MatMul
- Naive implementation
- SIMD optimization
- Benchmarking

**Days 3-4**: Memory optimization
- Tensor pooling
- KV-cache preallocation
- Profile memory usage

**Day 5**: Testing & validation
- End-to-end inference tests
- Numerical validation
- Performance baselines

### Week 3: Polish & Documentation

**Days 1-2**: SIMD hot paths
- Profile-guided optimization
- Hand-tune critical loops
- Benchmark improvements

**Days 3-4**: Integration testing
- Full model inference
- Compare with llama.cpp
- Stress testing

**Day 5**: Documentation
- Update Phase 10.8 summary
- Performance report
- Migration guide

## Success Criteria

### Minimum (Must Have)

- ✅ Q5_K dequantization working
- ✅ Tensor.At() supports all quant types
- ✅ End-to-end inference functional
- ✅ Numerical accuracy within 1e-3
- ✅ No crashes or memory leaks

### Target (Should Have)

- ✅ 5-10 tokens/second on CPU
- ✅ Memory usage <8GB for 14B model
- ✅ TTFT <1 second
- ✅ Quantized MatMul implemented
- ✅ Basic profiling complete

### Stretch (Nice to Have)

- ✅ 10-20 tokens/second
- ✅ Within 2x of llama.cpp
- ✅ SIMD optimization for dequant
- ✅ Memory pooling
- ✅ Flash Attention (basic)

## Risk Assessment

### High Risk
- **Quantization complexity**: Q5_K format may be tricky to implement correctly
  - **Mitigation**: Study llama.cpp thoroughly, use known-value tests

- **Performance gap**: Pure Go may be significantly slower than C++
  - **Mitigation**: Focus on SIMD, profiling, targeted optimization

### Medium Risk
- **Memory usage**: Dequantization may use too much memory
  - **Mitigation**: Implement smart caching, lazy dequant

- **Numerical accuracy**: Dequant errors could accumulate
  - **Mitigation**: Rigorous testing, comparison with llama.cpp

### Low Risk
- **API changes**: Minimal, mostly internal
- **Testing**: Good test infrastructure exists
- **Documentation**: Templates available from previous phases

## Performance Targets

### Model Loading
- ✅ Current: 106ms (excellent!)
- Target: Maintain <200ms

### Inference Speed
- Current: N/A (blocked)
- Target: 5-10 tok/s (14B model, CPU)
- Stretch: 10-20 tok/s

### Memory Usage
- ✅ Current: <2MB (mmap working)
- Target: <8GB total for 14B model
- Stretch: <6GB

### First Token Latency
- Current: N/A
- Target: <1 second
- Stretch: <500ms

## Profiling Plan

### Tools
- `go test -cpuprofile`
- `go test -memprofile`
- `go tool pprof`
- `perf` (Linux)
- `Instruments` (macOS)

### Focus Areas
1. Dequantization overhead
2. MatMul performance
3. Memory allocations
4. Cache efficiency

### Benchmarks
- Micro: Individual operations
- Component: Layer-wise
- End-to-end: Full generation

## Next Steps (Immediate)

1. **Start Task 1.1** - Study Q5_K format
   - Read llama.cpp `ggml-quants.c`
   - Document block structure
   - Create test cases

2. **Set up profiling** - While quantization is being implemented
   - Profile existing tensor operations
   - Identify current bottlenecks
   - Establish baselines

3. **Update plan** - As we learn more
   - Adjust timeline based on complexity
   - Reprioritize based on findings
   - Document lessons learned

## Open Questions

1. **Dequant strategy**: Eager vs lazy vs hybrid?
   - Answer: Start eager for embeddings, lazy for weights

2. **SIMD priority**: Hand-tuned vs compiler auto-vectorization?
   - Answer: Profile first, hand-tune only if needed

3. **Memory budget**: How much RAM is acceptable?
   - Answer: <8GB for 14B model (competitive with llama.cpp)

4. **Accuracy vs speed**: What tradeoffs are acceptable?
   - Answer: Must match llama.cpp within 1e-3

## Resources

### Documentation
- GGUF format spec
- llama.cpp quantization code
- GGML documentation
- Phase 10.1-10.7 summaries

### Tools
- Go profiler
- benchstat
- perf/Instruments
- Test GGUF models

### References
- llama.cpp: https://github.com/ggerganov/llama.cpp
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Quantization papers: Various (link in docs)

---

**Last Updated**: January 31, 2026
**Phase Status**: IN PROGRESS
**Current Task**: Setting up profiling and planning quantization implementation
**Next Milestone**: Q5_K dequantization working
