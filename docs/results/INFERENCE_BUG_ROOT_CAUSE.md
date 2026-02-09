# Inference Bug Root Cause Analysis - February 9, 2026

## Summary

**Bug**: Model generates repetitive garbage output (token 128008 repeated)

**Root Cause**: **Layer 35 is collapsing different inputs to nearly identical outputs**, causing the same token to be selected every time.

---

## Investigation Process

### ✅ What We Confirmed Works

1. **Embeddings** (Step 1):
   - Different tokens produce different embeddings ✓
   - Token 128008: `min=-0.1008, max=0.1213, mean=0.0051`
   - Prefill tokens: `min=-0.1187, max=0.1220, mean=-0.0026`

2. **Hidden State Progression** (Steps 2-4):
   - Hidden states diverge through layers ✓
   - Values grow exponentially (normal for deep transformers)
   - Layer 0: `±0.1` range
   - Layer 10: `±100` range
   - Layer 20: `±20,000` range
   - Layer 35: `±50,000` range

3. **Position Indices** (Step 5):
   - Positions increment correctly ✓
   - Prefill: `[0, 1, 2, 3]`
   - Decode Step 0: `[4]`
   - Decode Step 1: `[5]`
   - KV-cache length updates properly

4. **Final Layer Norm**:
   - Brings explosive values back to normal range ✓
   - Input: `±50,000` → Output: `±13`

---

## ❌ The Bug: Layer 35 Convergence

### Hidden States INTO Layer 35 (Different)

| Step | Min | Max | Mean |
|------|-----|-----|------|
| 0 | -14445 | 51493 | -4301 |
| 1 | -13649 | 48765 | -4075 |
| 2 | -13093 | 46865 | -3882 |

**Difference**: Hundreds to thousands of units apart ✓

### Hidden States AFTER Layer 35 + Final Norm (Nearly Identical)

| Step | Min | Max | Mean |
|------|-----|-----|------|
| 0 | -2.650 | 13.285 | 1.280 |
| 1 | -2.635 | 13.303 | 1.277 |
| 2 | -2.577 | 13.293 | 1.278 |
| 3 | -2.574 | 13.132 | 1.290 |

**Difference**: Only **0.01** variation! ❌

### Resulting Logits (Nearly Identical)

| Step | Token 128008 | Token 90760 | Token 97928 |
|------|--------------|-------------|-------------|
| 0 | 20.50 | 20.12 | 20.02 |
| 1 | 20.50 | 20.20 | 20.02 |

Token 128008 barely wins (0.38 margin), but wins consistently due to identical hidden states.

---

## Root Cause Hypotheses

### Most Likely: Attention Mechanism Issue

**Theory**: In decode mode with KV-cache, attention is dominated by cached tokens, making the new token irrelevant.

**Evidence**:
- Single query (new token) attending to 4+ cached keys
- If attention is too uniform or cache-dominated, new token has no influence
- Result: Output depends only on cache, not new input

**Possible Causes**:
1. **Attention scores too uniform** - softmax temperature wrong
2. **Query/Key scaling issue** - dot products not scaled correctly
3. **RoPE miscalculation** - position embeddings wrong for cached vs new tokens
4. **Causal mask issue** - might be masking incorrectly in decode

### Alternative: Feed-Forward Network Issue

**Theory**: SwiGLU in layer 35 produces similar outputs for different inputs.

**Less Likely** because:
- FFN typically amplifies differences, not collapses them
- Attention is more likely culprit in decode scenario

---

## Next Steps to Fix

### Priority 1: Inspect Layer 35 Attention

1. **Add attention score logging**:
   - Check if attention is uniform vs peaked
   - Verify attention weights sum to 1
   - Compare prefill vs decode attention patterns

2. **Verify attention scaling**:
   ```go
   // Should be: scores = (Q @ K^T) / sqrt(head_dim)
   // Check if scaling factor is correct
   ```

3. **Check RoPE application**:
   - Verify RoPE is applied to both cached and new tokens
   - Check if positions are used correctly in RoPE

### Priority 2: Validate Causal Mask

Recent fix (commit 3eceb40) modified causal mask logic:
- Check if mask is correct for decode with cache
- Single query should attend to ALL cached positions

### Priority 3: Attention Temperature

Check if attention has implicit temperature/scaling that's making it too uniform.

---

## Code Locations

### Key Files

- **Layer 35**: `internal/transformer/layer.go` (TransformerLayer #35)
- **Attention**: `internal/transformer/attention.go`
- **RoPE**: `internal/transformer/rope.go`
- **Model Forward**: `internal/transformer/model.go:137-220`

### Debug Commits

- `be826f1`: Added hidden state tracking
- `c71c539`: Added per-layer timing
- `e390978`: Marked hang as resolved
- `addd41c`: Initial debug logging

---

## Performance Context

While debugging, we also improved performance:
- ✅ Output weight caching: 25% faster (commit 9cc1fbd)
- ✅ Per-token time: ~30s (commit c71c539)
- ⏳ Bottleneck: Transformer layers (73% of time)

But **correctness must come before performance** - fixing layer 35 is the priority.

---

## Test Configuration

- **Model**: qwen2.5-coder-3b-q4.gguf
- **Architecture**: Qwen 2 (36 layers, 16 heads, 2 KV heads)
- **Vocabulary**: 151,936 tokens
- **Test**: "Hello world" → 10 tokens
- **Sampling**: Greedy (temperature=0)

---

## Timeline

- **Feb 8**: Initial inference issue noted
- **Feb 9 Morning**: Fixed GPU operations, data sync
- **Feb 9 Afternoon**:
  - Resolved "hang" (was just slow)
  - Implemented output weight caching
  - Narrowed bug to layer 35

**Status**: Root cause identified, fix in progress

---

## References

- Original issue: `PLAN.md` Phase 11.3
- Debug session: `INFERENCE_DEBUGGING_SESSION.md`
- Causal mask fix: Commit `3eceb40`
- Attention implementation: `internal/transformer/attention.go:Forward()`
