# Vibrant Inference Performance Optimization Plan

**Date**: February 1, 2026
**Status**: Analysis Complete - Ready for Implementation
**Priority**: CRITICAL - Blocking production-ready inference

---

## 🎯 Executive Summary

Based on comprehensive profiling and code analysis, we've identified **7 critical bottlenecks** causing:
- **8.6GB unnecessary allocations** per 32-layer inference pass
- **10-20% time wasted** on redundant operations
- **O(seq²) memory copying** during generation
- **640+ heap allocations** per forward pass causing GC pressure

**Estimated cumulative speedup**: 5-10x with quality preservation

---

## 📊 Profiling Results Summary

### Current Performance Baseline

**Tensor Operations** (from benchmarks):
- MatMul 256x256: 2.4ms (parallel), 12ms (naive) → **5x speedup achieved**
- Q5_K Dequantization: **303 MB/s** throughput
- Q6_K Dequantization: **314 MB/s** throughput
- Element access: 26ns per operation

**Memory Allocations**:
- Per forward pass (32 layers): **~8.4 MB**
- Total allocations per pass: **640+ tensors**
- Dequantization overhead: **8.6 GB** (entire model converted to Float32)

### Identified Bottlenecks (by impact)

| # | Bottleneck | Location | Impact | Estimated Speedup |
|---|------------|----------|--------|-------------------|
| 1 | Dequant + MatMul unfused | `matmul.go:28-45` | 8.6GB allocs | **3-5x** |
| 2 | Head transpose copies | `attention.go:198-239` | 10-20% time | **1.15x** |
| 3 | KV-cache concatenation | `attention.go:417-453` | O(seq²) | **5-10x decode** |
| 4 | Softmax 3-loop | `attention.go:375-410` | Cache misses | **3x** |
| 5 | Tensor accessor overhead | `tensor.go` | 500K+ checks | **1.5x** |
| 6 | 640+ allocations/pass | Multiple | GC pressure | **1.3x** |

**Cumulative speedup potential**: **5-10x** end-to-end

---

## 🔥 Critical Optimizations (Phase 1)

### 1. Fused Dequantize + MatMul ⚡ **HIGHEST IMPACT**

**Problem**: Currently dequantizes entire weight tensor before multiplication
```go
// Current (BAD): 8.6GB allocations for 32 layers
if b.dtype == Q5_K {
    bDequantized, err := DequantizeQ5_KTensor(b)  // Allocates 67-201MB per weight!
    b = bDequantized
}
return matmulSIMDParallel(a, b)
```

**Solution**: Dequantize blocks on-demand during multiplication
```go
// Proposed (GOOD): Zero intermediate allocations
func MatMulQ5K(a *Tensor, bQuantized *Tensor) *Tensor {
    for each output element C[i,j]:
        sum = 0
        for k in range:
            b_val = DequantizeQ5_KElement(bQuantized, k, j)  // On-demand
            sum += a.At(i, k) * b_val
        C[i,j] = sum
}
```

**Implementation**:
- Create `internal/tensor/matmul_quant.go`
- Implement `MatMulQ5K()`, `MatMulQ6K()`, `MatMulQ4K()`, `MatMulQ8_0()`
- Use block-wise dequantization for better cache locality
- Add SIMD optimizations for dequant loop

**Expected Results**:
- ✅ Memory: 50-80% reduction (8.6GB → <2GB)
- ✅ Speed: 2-3x faster (eliminates copy overhead + better cache)
- ✅ Quality: Identical (same dequant math)

**Estimated Time**: 4-5 days
**Priority**: CRITICAL

---

### 2. Eliminate Head Transpose 🚫 **HIGH IMPACT**

**Problem**: Copies all attention head data element-by-element
```go
// attention.go:198 - Wastes 10-20% inference time
func transposeHeads(x *Tensor) *Tensor {
    result := NewTensor([batch, heads, seq, headDim])
    for b, s, h, d:  // 4 nested loops
        result.Set(x.At(b, s, h, d), b, h, s, d)  // Bounds checks on every access
}
```

**Cost**:
- 1.6MB allocations per 32 layers
- 10-20% inference time on element-wise copying

**Solution**: Use correct tensor layout from start
```go
// Store Q, K, V directly as [batch, heads, seq, head_dim]
// No transpose needed!
q = MatMul(x, wq).Reshape([batch, heads, seq, head_dim])
```

**Implementation**:
- Modify `attention.go` projection reshaping
- Remove `transposeHeads()` and `untransposeHeads()` functions
- Update all attention operations for new layout

**Expected Results**:
- ✅ Speed: 10-20% faster (eliminate transpose)
- ✅ Memory: 1.6MB saved
- ✅ Quality: Identical

**Estimated Time**: 2-3 days
**Priority**: HIGH

---

### 3. Ring Buffer KV-Cache 🔄 **HIGH IMPACT**

**Problem**: O(seq²) copying during generation
```go
// attention.go:417 - Copies entire cache every decode step
func concatenateSeqDim(cached, new) {
    result := NewTensor([..., cachedLen + newLen, ...])
    // Copy all cached values (grows each step!)
    for b, h, s, d:
        result.Set(cached.At(b, h, s, d), ...)
    // Copy new values
    for b, h, s, d:
        result.Set(new.At(b, h, s, d), ...)
}
```

**Cost**: For seq_len=512: `1 + 2 + 3 + ... + 512 = 131K` element copies

**Solution**: Pre-allocated circular buffer
```go
type RingBufferCache struct {
    buffer []*Tensor  // Pre-allocated to max_seq_len
    head   int        // Current write position
    len    int        // Current cache length
}

func (c *RingBufferCache) Update(new *Tensor) {
    // Copy new values to buffer[head]
    // Increment head (mod max_len)
}
```

**Implementation**:
- Create `RingBufferCache` in `internal/inference/cache.go`
- Update `attention.go` to use ring buffer
- Pre-allocate to max context length

**Expected Results**:
- ✅ Speed: 5-10x faster decode (O(1) vs O(seq))
- ✅ Memory: Zero allocations during decode
- ✅ Quality: Identical

**Estimated Time**: 3-4 days
**Priority**: HIGH

---

## ⚙️ Important Optimizations (Phase 2)

### 4. Fused Single-Pass Softmax

**Current**: 3 separate loops + per-row allocation
- Loop 1: Find max
- Loop 2: Compute exp (allocates array)
- Loop 3: Normalize

**Proposed**: Single fused loop
```go
func softmaxFused(scores []float32) {
    // Single pass: max + exp + sum + normalize
    max := findMaxSIMD(scores)
    sum := 0.0
    for i := range scores {
        scores[i] = exp(scores[i] - max)
        sum += scores[i]
    }
    scale := 1.0 / sum
    for i := range scores {
        scores[i] *= scale
    }
}
```

**Expected**: 3x faster, zero allocations

**Estimated Time**: 2 days
**Priority**: MEDIUM

---

### 5. Tensor Memory Pooling

**Current**: 640+ allocations per forward pass

**Proposed**: Reuse tensor objects via sync.Pool
```go
var tensorPool = sync.Pool{
    New: func() interface{} {
        return &Tensor{}
    },
}

func getTensor(shape []int, dtype DataType) *Tensor {
    t := tensorPool.Get().(*Tensor)
    t.Reshape(shape)
    return t
}
```

**Expected**: 50-100 allocations per pass, lower GC pressure

**Estimated Time**: 4-5 days
**Priority**: MEDIUM

---

### 6. SIMD Dequantization

**Current**: 303-314 MB/s (scalar)
**Target**: 900-1200 MB/s (vectorized)

**Implementation**:
- AVX2 for x86_64
- NEON for ARM64
- Vectorize bit unpacking + scale application

**Expected**: 3-4x dequantization throughput

**Estimated Time**: 5-6 days
**Priority**: MEDIUM

---

## 📋 Implementation Roadmap

### Week 1: Critical Path (Blocking Performance)
**Days 1-3**: Fused Dequant + MatMul
- Implement MatMulQ5K, MatMulQ6K
- Benchmark vs current
- Integrate into transformer

**Days 4-5**: Eliminate Head Transpose
- Update attention tensor layout
- Remove transpose functions
- Validate correctness

### Week 2: High-Impact Optimizations
**Days 1-2**: Ring Buffer KV-Cache
- Implement RingBufferCache
- Update attention.go
- Test with long sequences

**Days 3-4**: Fused Softmax
- Single-pass implementation
- SIMD vectorization
- Benchmark

**Day 5**: Integration Testing
- End-to-end inference
- Measure actual tokens/sec
- Validate quality

### Week 3: Memory & Polish
**Days 1-3**: Tensor Memory Pooling
- Implement TensorPool
- Integrate into all layers
- Measure GC improvement

**Days 4-5**: SIMD Dequantization
- AVX2/NEON implementations
- Benchmark improvements
- Cross-platform testing

---

## 📊 Success Metrics

| Metric | Current | Target (Phase 1) | Target (All) |
|--------|---------|------------------|--------------|
| **Tokens/sec (14B)** | Unknown | 10-15 | 20-30 |
| **Memory (inference)** | 8.6GB | <2GB | <2GB |
| **Allocations/pass** | 640+ | 200 | <50 |
| **TTFT** | Unknown | <1s | <500ms |
| **Dequant throughput** | 303 MB/s | 300 MB/s | 1000 MB/s |
| **Quality loss** | N/A | **0%** | **0%** |

---

## ✅ Quality Preservation Strategy

**All optimizations must maintain numerical accuracy**:

1. **Validation tests**: Compare outputs with baseline implementation
   - Max difference: < 1e-4 for Float32 operations
   - Bit-exact for deterministic operations

2. **Reference outputs**: Generate with current implementation
   - Save intermediate tensors at each layer
   - Compare optimized vs reference

3. **End-to-end testing**:
   - Generate 100+ tokens with same seed
   - Compare text outputs (should be identical)
   - Compare logits (within 1e-4)

4. **Continuous benchmarking**:
   - Every PR must include benchmark comparison
   - No regression on quality metrics

---

## 🎯 Expected Final Results

**After Phase 1** (Week 1-2):
- 5-8x faster inference
- 80% memory reduction
- Zero quality loss
- Tokens/sec: 10-15 for 14B model

**After All Phases** (Week 1-3):
- 8-12x faster inference
- 90% memory reduction
- <50 allocations per pass
- Tokens/sec: 20-30 for 14B model
- Production-ready performance

---

## 📝 Next Steps

1. ✅ **Analysis complete** - This document
2. ⏳ **Start implementation** - Fused Dequant+MatMul (highest impact)
3. ⏳ **Continuous validation** - Quality tests after each optimization
4. ⏳ **Benchmark tracking** - Measure cumulative improvements
5. ⏳ **Documentation** - Update specs as we optimize

---

**Ready to proceed with implementation.**
