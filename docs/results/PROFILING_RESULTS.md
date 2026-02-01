# Deep Profiling Results - Vibrant Inference Performance

## Executive Summary

**Profiling Date**: 2026-02-01  
**Model**: Qwen 2.5 Coder 7B Q4_K_M (4.2GB)  
**Test**: Single forward pass with 3 tokens  
**Total Time**: 99.13 seconds  
**Performance**: ~0.03 tokens/sec (need 20-30 tok/sec = **600-1000x speedup needed**)

## 🔥 Critical Bottleneck Found

**Matrix transpose loop consumes 75% of inference time!**

Location: `internal/tensor/matmul_simd.go:116-117`

```go
// THIS IS THE BOTTLENECK - 74.87s (71% of total time!)
for k := 0; k < K; k++ {
    for j := 0; j < N; j++ {
        bTransposed[j*K+k] = bData[k*N+j]  // Cache-unfriendly random access
    }
}
```

## Performance Breakdown

### Top Time Consumers

| Function | Time | % | Issue |
|----------|------|---|-------|
| **Matrix transpose** | **~75s** | **71%** | Cache-unfriendly loop |
| Matrix dot products | ~11s | 10% | Reasonable |
| Q4_K dequantization | ~12s | 11% | Per-call overhead |
| Memory allocation | ~6s | 6% | Excessive allocs |
| Other | ~2s | 2% | Various |

### Root Causes

**1. Naive Matrix Transpose (71%)**  
- Strided memory access (cache misses)
- Repeated for **every matmul call** (168-224x per forward pass!)
- Weight matrices are static - should transpose once at load time

**2. Redundant Operations**  
- Transposes same weight matrices repeatedly
- Should: Transpose once during loading → Store → Reuse

**3. Q4_K Dequantization (11%)**  
- Dequantizes same weights repeatedly
- Should: Dequantize once → Cache → Reuse

## Quick Win Recommendations

### ✅ Priority 1: Pre-transpose Weights (71% speedup potential)

Transpose weight matrices **once** during model loading:

```go
// In transformer.NewModel():
for _, layer := range model.layers {
    layer.attention.wq = layer.attention.wq.Transpose()
    layer.attention.wk = layer.attention.wk.Transpose()
    // ... transpose all weight matrices
}
```

Then skip runtime transpose in matmul.

**Expected**: 99s → ~25s (**4x speedup**)  
**Effort**: 2-3 hours  
**Risk**: Low (just reordering existing code)

### ✅ Priority 2: Cache Dequantized Weights (11% speedup)

Dequantize Q4_K/Q6_K weights once and cache:

```go
// Lazy dequantization with caching
type Model struct {
    dequantCache map[string]*Tensor
}
```

**Expected**: 25s → 20s (additional 20% speedup)  
**Memory**: 4.2GB → ~16GB (acceptable on 31GB system)

### ✅ Priority 3: Reduce Allocations (5% speedup)

Preallocate activation buffers, reuse across layers.

**Expected**: 20s → 18s (additional 10% speedup)

## Combined Expected Performance

- **Current**: 99s per forward pass (0.03 tok/sec)
- **After all optimizations**: 18-20s (0.15-0.20 tok/sec)
- **Speedup**: ~5x

Still need another 25-50x for production speed, but this is a solid foundation!

## Next Steps

1. **Implement pre-transpose** (Priority 1) - biggest impact
2. **Profile again** to verify improvement  
3. **Cache dequantized weights** (Priority 2)
4. **Investigate attention optimizations** (Flash Attention, etc.)
