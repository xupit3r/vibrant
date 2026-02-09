# Inference Debugging Session - February 9, 2026

## Session Goal
Investigate and fix the core inference issue where the model produces repetitive garbage output.

## Issues Discovered

### 1. Missing GPU Operations (Fixed ✅)
**Commits**: `6365346`, `28779e8`

**Problem**: Compilation errors when building with GPU support:
- `undefined: siluGPU`
- `undefined: tensor.RMSNormGPU`
- `undefined: tensor.RoPEGPU`

**Solution**:
- Added missing `siluGPU()` implementation in `internal/tensor/ops_gpu.go`
- Added public GPU wrappers: `RMSNormGPU`, `RoPEGPU`, `SoftmaxGPU`
- `RoPEGPU` returns nil (not implemented yet, falls back to CPU)

**Status**: Fixed and committed

---

### 2. GPU Data Sync Bug (Fixed ✅)
**Commit**: `fc55006`

**Problem**: `extractLastTokenLogits()` used `Data()` which doesn't sync from GPU:
```go
src := logits.Data().([]float32)  // Returns stale data if tensor is on GPU!
```

**Solution**: Use `EnsureCPUData()` to properly sync from GPU:
```go
srcData, err := logits.EnsureCPUData()
if err != nil {
    srcData = logits.Data()  // Fallback
}
src := srcData.([]float32)
```

**Impact**: Critical for GPU inference where logits are computed on GPU but sampled on CPU.

**Additional Issues Found** (not yet fixed):
- `At()` and `Set()` methods also use `Data()` directly without syncing
- These need fixing for safe GPU tensor access

**Status**: Fixed for `extractLastTokenLogits`, broader fix needed

---

### 3. Q6_K Performance Issue (Diagnosed ⚠️)
**Commit**: `addd41c` (debug logging)

**Problem**: Inference appears to "hang" but is actually just extremely slow.

**Root Cause**:
- Output weight matrix is **Q6_K quantized**: `[2048 x 151936]` = 311M elements
- Each forward pass dequantizes on-the-fly during MatMul
- **~10-15 seconds per token** on CPU!

**Evidence**:
```
[MODEL] Output weight shape: [2048 151936], dtype: q6_k
[MODEL] Starting MatMul (this may take a while for large vocab)...
[MODEL] MatMul complete  ← Takes 10-15 seconds!
```

**Why It's Slow**:
1. Vocabulary size: 151,936 tokens (Qwen 2.5 has massive vocab)
2. Matrix multiplication: `[1 x 2048] @ [2048 x 151936]` = 311M multiplications per token
3. Q6_K dequantization: Every element must be dequantized from 6.5 bits to float32
4. No caching: Output weight is dequantized fresh every forward pass

**Potential Solutions**:
1. **Cache dequantized output weight** (1.2GB of float32, but worth it)
2. **Use GPU** (6.4x speedup for large matmuls)
3. **Implement fused Q6_K MatMul** (avoid intermediate dequantization)
4. **Vocabulary pruning** (unlikely - breaks the model)

**Status**: Diagnosed, not yet fixed

---

### 4. Repetitive Garbage Output (Unsolved ❌)
**Commits**: Ongoing investigation since `3eceb40`

**Problem**: Model produces the same token repeatedly regardless of input.

**Symptom**:
```
Prompt: "Hello world"
Output: "íİĺìĿ´ì§Ģ" (token 128008) × 10

[DEBUG] Step 0: sampled token 128008
[DEBUG] Logits: top-5: (128008:20.50) (90760:20.12) (97928:20.02)
[DEBUG] Step 1: sampled token 128008
[DEBUG] Logits: top-5: (128008:20.50) (90760:20.12) (97928:20.02)  ← Nearly identical!
[DEBUG] Step 2: sampled token 128008
[DEBUG] Logits: top-5: (128008:20.50) (90760:20.20) (97928:20.02)
```

**Characteristics**:
- Logits barely change between decode steps (min/max shift by ~0.05)
- Same top-5 tokens in nearly same order
- Happens on both CPU and GPU (not device-specific)
- Affects all prompts (not prompt-specific)

**What Works**:
- ✅ Model loads successfully
- ✅ Embeddings complete
- ✅ All 36 layers execute
- ✅ Output norm completes
- ✅ Output projection completes
- ✅ Logits have reasonable shape: `[1, seq_len, 151936]`
- ✅ No crashes or errors

**What Doesn't Work**:
- ❌ Hidden states not varying enough between steps
- ❌ KV-cache might not be working correctly
- ❌ Position embeddings might be wrong
- ❌ Output projection might be corrupted

**Investigation Status**:
- Recent fix (commit `3eceb40`) addressed causal mask and position indices
- Commit message noted: "model generates varying logits" but still wrong token
- Currently: Logits NOT varying enough

**Next Steps**:
1. Verify embeddings produce different vectors for different tokens
2. Check if hidden states change through layers
3. Inspect KV-cache contents (are cached keys/values correct?)
4. Verify position indices are incrementing correctly
5. Check if output weight dequantization is correct (Q6_K validation)
6. Compare logits to known-good implementation (llama.cpp)

**Status**: Under investigation

---

## Test Configuration

**Model**: `qwen2.5-coder-3b-q4.gguf`
- Architecture: Qwen 2
- Context: 32,768 tokens
- Vocabulary: 151,936 tokens (unusually large!)
- Dimensions: 2048
- Layers: 36
- Heads: 16 (2 KV heads - grouped query attention)
- FFN: 11,008

**Test Setup**:
- Device: CPU
- Temperature: 0.0 (greedy sampling)
- Max tokens: 10
- Debug logging: Enabled

---

## Timeline

- **Feb 8, 2026**: Causal mask fix (commit `3eceb40`) - model generated tokens but wrong ones
- **Feb 9, 2026**:
  - Fixed missing GPU operations
  - Fixed GPU data sync bug
  - Discovered Q6_K performance issue
  - Confirmed repetitive output persists

---

## Key Code Locations

- **Inference engine**: `internal/inference/engine.go`
- **Model forward pass**: `internal/transformer/model.go`
- **Transformer layers**: `internal/transformer/layer.go`
- **Attention**: `internal/transformer/attention.go`
- **GPU operations**: `internal/tensor/ops_gpu.go`
- **Tensor data access**: `internal/tensor/tensor.go` (lines 417, 449, 475)

---

## Performance Baseline

**Current Performance** (CPU, Q6_K):
- Prefill (4 tokens): ~40-50 seconds
- Decode (per token): ~10-15 seconds
- Total for 10 tokens: ~2-3 minutes

**Expected Performance** (target):
- Prefill: <5 seconds
- Decode: <1 second per token
- Total for 10 tokens: <15 seconds

**Gap**: 10-20x slower than expected

---

## Open Questions

1. Why do logits barely change between decode steps?
2. Is the KV-cache actually being used?
3. Is Q6_K dequantization producing correct values?
4. Why does token 128008 consistently score highest?
5. Should we cache the dequantized output weight?
6. Is the grouped query attention (16 heads, 2 KV heads) implemented correctly?

---

## References

- Original issue: `PLAN.md` - Phase 11.3 notes
- Causal mask fix: Commit `3eceb40`
- GPU validation: `docs/results/GPU_VALIDATION_RESULTS.md`
- Q6_K implementation: `docs/implementation/Q6K_IMPLEMENTATION.md`
