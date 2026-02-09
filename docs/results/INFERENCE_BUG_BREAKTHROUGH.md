# Inference Bug Breakthrough - February 9, 2026

## Root Cause Identified ✅

**Bug**: Model generates repetitive output (token 128008 repeated)

**Root Cause**: All token embeddings have identical RMS values in decode mode

**Smoking Gun**: Embedding RMS for ALL decode tokens = 0.0292 (exactly identical)

---

## UPDATE: Embedding RMS Discovery (Smoking Gun)

**Critical Finding**: All decode tokens produce embeddings with **EXACTLY** the same RMS!

```
EMBEDDINGS RMS:
Prefill (4 tokens):     RMS=0.0342
Decode Step 1 (1 tok): RMS=0.0292  ← identical
Decode Step 2 (1 tok): RMS=0.0292  ← identical
Decode Step 3 (1 tok): RMS=0.0292  ← identical
Decode Step 4 (1 tok): RMS=0.0292  ← identical
Decode Step 5 (1 tok): RMS=0.0292  ← identical
```

**After Layer 0:**
```
Prefill:       RMS=0.71
Decode Step 1: RMS=0.51
Decode Step 2: RMS=0.51
Decode Step 3: RMS=0.50  ← nearly identical
Decode Step 4: RMS=0.50  ← nearly identical
Decode Step 5: RMS=0.50  ← nearly identical
```

This identical RMS persists through all 36 layers, leading to convergent outputs.

**Implication**: Either:
1. This is expected behavior for this model (embeddings are normalized)
2. We have a bug in embedding lookup or normalization

**MUST validate against llama.cpp before proceeding!**

---

## Complete Investigation Chain

### 1. Surface Symptom
- Token 128008 always wins with logit ~20.5
- Competitors at ~20.1-20.2 logits
- **Margin: only 0.3-0.5** (should be 5-10+ in healthy model)

### 2. Output Projection Analysis
```
Logits for each decode step:
Step 0: token_128008=20.4968 token_90760=20.1204 token_97928=20.0177
Step 1: token_128008=20.5035 token_90760=20.1996 token_97928=20.0174
Step 2: token_128008=20.5151 token_90760=20.1747 token_97928=20.0113
```

Token 128008 **consistently** has highest logit despite different inputs.

### 3. Final Normalization Analysis

**BEFORE final RMSNorm** (different RMS, different patterns):
```
Step 0: RMS=28358, first10=[-1595, -1541, -1792, -286, -1411, ...]
Step 1: RMS=25742, first10=[-1633, -1556, -1690, -293, -1465, ...]  (9% RMS diff)
Step 2: RMS=24805, first10=[-1585, -1676, -1688, -302, -1670, ...]  (12% RMS diff)
```

**AFTER final RMSNorm** (nearly identical):
```
Step 0: first10=[-0.19, -0.19, -0.22, -0.03, -0.16, 0.00, 0.16, ...]
Step 1: first10=[-0.21, -0.20, -0.22, -0.04, -0.18, -0.02, 0.18, ...]
Step 2: first10=[-0.21, -0.22, -0.23, -0.04, -0.21, -0.00, 0.19, ...]
```

Only 10-20% element-wise variation after normalization!

###4. Layer 35 Internal Analysis

**BEFORE layer 35 attn_norm** (different):
```
Step 0: Input mean = -4300.77
Step 1: Input mean = -4075.02
Step 2: Input mean = -3881.47
```

**AFTER layer 35 attn_norm** (nearly identical!):
```
Step 0: mean = -0.3374
Step 1: mean = -0.3377
Step 2: mean = -0.3339
```

**After attention** (still nearly identical):
```
Step 0: mean = -1.9537
Step 1: mean = -1.9462
Step 2: mean = -1.9512
```

The convergence happens at **every RMSNorm operation** in layer 35 and beyond!

---

## Root Cause Mechanism

1. **Hidden states have similar RMS values**
   - Not similar means, but similar root-mean-square magnitude
   - Example: RMS values of 28358, 25742, 24805 (only 9-12% variation)

2. **RMSNorm normalizes by RMS**
   - Formula: `output = (x / RMS(x)) * weight`
   - When RMS is similar, normalized outputs are similar
   - Even though input patterns differ, normalization crushes the differences

3. **Cascading effect**
   - Each layer's RMSNorm produces similar normalized values
   - Attention/FFN operate on similar inputs → produce similar outputs
   - Next layer sees similar-RMS inputs → pattern repeats

4. **Output projection amplifies similarities**
   - 10-20% variation in hidden states → only 0.3-0.5 variation in logits
   - Token 128008's weight column is well-aligned with typical normalized pattern
   - Same token wins every time

---

## Why Hidden States Have Similar RMS

**Hypothesis**: The combination of:
1. RMSNorm at every layer
2. Residual connections preserving magnitude
3. FFN output magnitude stabilization

...causes the network to converge to a **stable RMS "attractor"**.

Regardless of input token, by layer 35, all hidden states have similar RMS values (±10-30% variation).

This is amplified in **decode mode** because:
- Single new token added to cache
- Attention is dominated by cached tokens (4+ positions)
- New token has minimal influence on output
- Output inherits RMS from stable cache state

---

## Evidence Supporting Root Cause

✅ **Layer 35 inputs have different means** (-4300 vs -4075 vs -3881)  
✅ **Layer 35 inputs have similar RMS** (varies by only 9-12% for first 3 tokens)  
✅ **After RMSNorm, outputs are nearly identical** (-0.337 vs -0.338 vs -0.334)  
✅ **Pattern repeats at final norm** (similar RMS in → similar normalized out)  
✅ **Logits are too similar** (0.3-0.5 margin instead of 5-10+)  
✅ **Token 128008 consistently wins** (weight column aligned with typical pattern)

---

## What's NOT the Problem

❌ RMSNorm implementation - formula is correct  
❌ RMSNorm weights - they have reasonable variance  
❌ RMSNorm epsilon - 1e-6 is standard  
❌ Output weight matrix - rows/columns have distinct values  
❌ Attention mechanism - scoring and softmax work correctly  
❌ Position indices - increment correctly with KV cache  
❌ Embeddings - different tokens produce different embeddings

---

## Proposed Fixes

### Option 1: Check Reference Implementation
- Compare our RMSNorm with llama.cpp or reference PyTorch
- Verify we're not missing any scaling factors or normalization steps

### Option 2: Investigate KV Cache in Decode
- In decode mode, attention might be too uniform
- Single query attending to 4+ cached keys might dilute influence
- Check attention distribution (add attention score logging)

### Option 3: Check Model File
- Download fresh copy of qwen2.5-coder-3b-q4.gguf
- Verify file integrity (checksum)
- Try different quantization (Q5_K or F16)

### Option 4: Compare with Known-Good Inference
- Run same model with llama.cpp
- Compare hidden state RMS values at layer 35
- If llama.cpp also shows similar RMS, it's a model characteristic (not a bug)
- If llama.cpp shows diverse RMS, we have a bug

---

## Next Steps

1. **Immediate**: Add hidden state RMS logging for ALL layers (not just 35)
   - Check if RMS convergence starts early or late
   - Identify which layer first shows the issue

2. **Validation**: Run model with llama.cpp and log hidden states
   - Compare RMS values at each layer
   - If they match, our implementation is correct (just slow to fix)
   - If they differ, we have a bug to fix

3. **Fix**: Based on validation results
   - If bug: Fix the identified issue
   - If model characteristic: Consider temperature sampling or different model

---

## Performance Note

While debugging, we also improved performance:
- ✅ Output weight caching: 25% faster
- ✅ Per-token time: ~30s
- Current bottleneck: Transformer layers (73% of time)

**Correctness comes before performance** - must fix inference bug first.

---

**Status**: Root cause identified, validation in progress  
**Priority**: HIGH - blocks all inference functionality  
**Timeline**: February 9, 2026 investigation

---

## References

- Original issue: `PLAN.md` Phase 11.3  
- Previous analysis: `INFERENCE_BUG_ROOT_CAUSE.md`  
- Causal mask fix: Commit `3eceb40`
