# Pre-Transpose Optimization Benchmark Results

**Date**: February 2, 2026
**Model**: qwen2.5-coder-7b-q4.gguf (4.7GB)
**Test**: Forward pass performance with pre-transposed weights

## Executive Summary

Benchmarking revealed that the pre-transpose optimization **does NOT achieve the expected 4x speedup** due to **cache thrashing**. The optimization is implemented correctly, but insufficient cache capacity (8GB vs 26GB needed) causes constant re-transposition of weights.

**Actual Results:**
- **Forward pass time**: 64-70 seconds (vs expected ~25s)
- **Actual speedup**: 1.4-1.5x (vs expected 4x)
- **Transpose still consuming**: 43% of CPU time (should be ~0%)

## Performance Measurements

### Expected vs Actual

| Metric | Expected (Phase 10.9) | Actual (Measured) | Delta |
|--------|----------------------|-------------------|-------|
| **Before optimization** | 99.13s | ~99s (baseline) | - |
| **After optimization** | ~25s (4x speedup) | 64-70s | 2.6-2.8x slower |
| **Transpose time** | ~0s (eliminated) | 38.90s (43%) | Still dominant! |
| **Transpose operations** | 0 (cached) | Many (cache misses) | Not eliminated |
| **Cache hit rate** | ~100% | ~23% (46/200) | Cache thrashing |

### CPU Profile Breakdown

```
Function                    Time      % of Total    Notes
-----------------------------------------------------------------
Transpose                  38.90s       43.48%     ← Still #1 bottleneck!
DequantizeQ4_K            10.50s       11.74%     Cache misses
vectorDotProduct          10.36s       11.58%     Actual computation
DequantizeQ6_K             6.59s        7.37%     Cache misses
RoPE (sin/cos)            ~15.0s       ~17%       Expected
Other operations           ~8.0s        ~9%       Expected
-----------------------------------------------------------------
Total                     89.46s       100%
```

**Key Observation**: Transpose is still the dominant operation (43%), indicating the optimization is not working as intended.

## Root Cause: Cache Thrashing

### Memory Requirements

| Component | Size | Notes |
|-----------|------|-------|
| **Model (compressed)** | 4.7 GB | Q4_K quantized weights |
| **Dequantized weights** | 26.34 GB | All weights as Float32 |
| **Cache budget** | 8 GB | Default WeightCacheManager limit |
| **Available RAM** | 27 GB | System total |
| **Problem** | 8 GB << 26 GB | Cache can't hold all weights |

### Cache Behavior

**What Happens:**
1. First 46 weight tensors fit in 8GB cache (dequantized + transposed)
2. Forward pass needs more weights → evicts old weights (LRU)
3. Next forward pass needs evicted weights → re-dequantize + re-transpose
4. **Result**: Constant eviction and re-creation = transposes still happening

**Cache Statistics** (from benchmark):
- Cache used: 8190 MB / 8192 MB (99.9% full)
- Cached tensors: 46 out of ~200 weights
- Cache hit rate: ~23%
- **Conclusion**: Severe cache thrashing

### Why This Wasn't Caught

The pre-transpose optimization **is implemented correctly**:
- ✅ Weights are transposed during loading
- ✅ Transposed flag is set properly
- ✅ GetOrDequantTranspose() checks the flag
- ✅ MatMul skips transpose for flagged tensors

**The issue**: The cache eviction strategy doesn't account for:
- Layer-wise access patterns (weights used sequentially by layer)
- Memory capacity constraints (26GB > 8GB)
- Re-transpose cost when weights are evicted

## Bug Found & Fixed

### Slice Bounds Error

**Location**: `internal/tensor/matmul_simd.go` (all 4 functions)

**Problem**:
```go
// Before fix - WRONG for transposed tensors!
M, K := a.shape[0], a.shape[1]
N := b.shape[1]  // If B is transposed [N,K], this returns K!
```

When B is pre-transposed:
- Original shape: `[K, N]`
- Transposed shape: `[N, K]`
- `b.shape[1]` returns `K`, not `N` → **wrong dimension!**

**Solution**:
```go
// After fix - checks transpose flag
M, K := a.shape[0], a.shape[1]
N := b.shape[1]
if b.IsTransposed() {
    N = b.shape[0]  // Read N from first dimension
}
```

**Impact**:
- Fixed slice bounds panics
- Correct output dimensions
- All benchmarks now run successfully

**Applied to**:
- `matmulSIMD()`
- `matmulSIMDBlocked()`
- `matmulSIMDSingleRow()`
- `matmulSIMDParallel()`

## Benchmark Infrastructure

### Test Setup

**Hardware**:
- CPU: 12th Gen Intel Core i5-1240P (12 cores)
- RAM: 27 GB available
- OS: Linux 6.18.7-arch1-1

**Model**:
- qwen2.5-coder-7b-q4.gguf
- Size: 4.7 GB (compressed)
- Weights: 26.34 GB (dequantized Float32)
- Layers: 28

**Cache Configuration**:
- Initial tests: 8 GB budget (default)
- Enhanced tests: 20 GB budget
- Eviction: LRU policy

### Benchmarks Created

**1. forward_bench_test.go** (enhanced)
```go
BenchmarkForwardPass-12    1    64.2s/op
```
- Full forward pass timing
- Cache statistics logging
- Warmup pass for cache priming

**2. test_pretranspose.go** (new)
- Standalone cold vs warm cache test
- Measures 3 consecutive passes
- Reports cache growth and speedup

### How to Run

```bash
# Standard benchmark
go test ./internal/transformer/ -bench=BenchmarkForwardPass -timeout=10m

# With CPU profiling
go test ./internal/transformer/ -bench=BenchmarkForwardPass \
    -cpuprofile=cpu.prof -timeout=10m

# View profile
go tool pprof -http=:8080 cpu.prof

# Standalone test
go run test_pretranspose.go
```

## Solutions (Recommendations)

### Option 1: Layer-wise Cache Management ⭐ RECOMMENDED

**Strategy**: Cache only current layer's weights during forward pass.

**Rationale**:
- Transformer processes layers sequentially
- Each layer needs ~1-2 GB of weights
- 8GB cache can hold 4-8 layers comfortably
- Evict previous layer after processing

**Expected Results**:
- Cache hit rate: ~100% (no thrashing)
- Forward pass time: ~25s (4x speedup achieved)
- Memory: 8GB cache sufficient

**Implementation**:
```go
type LayerWiseCache struct {
    currentLayer int
    maxLayers    int
    cache        map[int][]*Tensor
}

func (c *LayerWiseCache) BeforeLayer(layer int) {
    if layer > 0 {
        // Evict previous layer's weights
        delete(c.cache, layer-1)
    }
}
```

**Effort**: 1-2 days

---

### Option 2: On-Disk Pre-transpose

**Strategy**: Transpose weights once during model loading, store in mmap format.

**Rationale**:
- Transpose happens once, stored permanently
- No runtime transpose ever needed
- No cache dependency

**Expected Results**:
- Forward pass time: ~20-25s (4-5x speedup)
- No cache required
- Slight increase in model file size

**Implementation**:
- Modify GGUF loader to transpose during parsing
- Store transposed layout in memory-mapped file
- Requires GGUF format changes

**Effort**: 3-5 days

---

### Option 3: Fused Dequant+Transpose Kernel

**Strategy**: Combine dequantization and transpose in single operation.

**Rationale**:
- Better cache locality (process data once)
- Eliminate separate transpose pass
- Optimize memory access patterns

**Expected Results**:
- Forward pass time: ~15-20s (5-6x speedup)
- Better cache utilization
- More complex to implement

**Implementation**:
- Create fused kernel for each quant type
- SIMD optimization for performance
- Extensive testing for correctness

**Effort**: 1-2 weeks

## Lessons Learned

### 1. Cache Capacity Matters

**Assumption**: 8GB cache is sufficient for weight caching
**Reality**: 26GB needed for full model → severe cache thrashing
**Lesson**: Always verify cache requirements match available memory

### 2. Implementation ≠ Performance

**Fact**: Pre-transpose optimization is correctly implemented
**But**: Doesn't deliver expected speedup due to cache limitations
**Lesson**: Correct code doesn't guarantee good performance

### 3. Profile Before and After

**Profiling revealed**:
- Transpose still consuming 43% of time (vs expected ~0%)
- Cache thrashing as root cause
- Need for layer-wise caching strategy

**Lesson**: Always measure actual performance, don't assume

### 4. Bug Discovery Through Benchmarking

**Bug found**: Slice bounds error in matmul_simd.go
**Found by**: Running benchmarks with real model
**Impact**: Would have caused panics in production

**Lesson**: Real-world testing catches bugs that unit tests miss

## Next Steps

### Immediate (Required for 4x Speedup)

1. **Implement Layer-wise Cache Management** (Option 1)
   - Modify WeightCacheManager to track layer ownership
   - Add layer begin/end hooks to transformer
   - Evict previous layer's weights after processing

2. **Re-benchmark with Layer-wise Caching**
   - Verify 4x speedup is achieved
   - Confirm transpose operations eliminated
   - Validate cache hit rate ~100%

3. **Update Documentation**
   - Document cache requirements
   - Add layer-wise caching strategy
   - Update PHASE10.9 summary with findings

### Short-term (Optimizations)

4. **Priority 2: Enhanced Weight Caching**
   - Smart eviction based on access patterns
   - Pre-load next layer's weights
   - Optimize cache memory layout

5. **Priority 3: Reduce Allocations**
   - Preallocate activation buffers
   - Reuse tensors across layers
   - Memory pooling

### Long-term (Research)

6. **Explore Option 2 or 3**
   - On-disk pre-transpose for permanent solution
   - Fused kernel for maximum performance
   - Hybrid approach (cache + on-disk)

## Conclusion

The pre-transpose optimization is **correctly implemented** but **doesn't achieve expected performance** due to cache thrashing. The solution is straightforward: implement layer-wise cache management to ensure weights stay cached during layer processing.

**Key Takeaways**:
- ✅ Optimization logic is sound
- ✅ Bug fixed (slice bounds error)
- ✅ Benchmark infrastructure working
- ❌ Cache capacity insufficient (8GB vs 26GB)
- ❌ Need layer-wise caching for 4x speedup
- 🎯 Clear path forward (Option 1)

**Status**: Blocked on cache management improvements
**Next Phase**: Implement layer-wise caching (Phase 10.10)
**Expected Outcome**: 4x speedup achievable with proper caching strategy

---

**Last Updated**: February 2, 2026
**Benchmark Model**: qwen2.5-coder-7b-q4.gguf
**Cache Budget Tested**: 8GB (default), 20GB (enhanced)
**Commits**: 4797248 (bug fix), 7870535 (benchmarks)
