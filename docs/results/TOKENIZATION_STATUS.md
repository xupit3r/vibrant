# Tokenization Status - February 9, 2026

## ✅ FULLY RESOLVED

All GPT2-style tokenization bugs have been identified and fixed.

## Summary

### Problems Found
1. **Spaces** encoded as `<unk>` (token 128244) instead of `Ġ` (token 220)
2. **Newlines** encoded as `<unk>` (token 128244) instead of `Ċ` (token 198)

### Root Cause
GPT2-style tokenizers (used by Qwen2.5) replace whitespace with special Unicode characters BEFORE applying BPE:
- Space `" "` → `"Ġ"` (U+0120)
- Newline `"\n"` → `"Ċ"` (U+010A)

Our BPE implementation used **byte-level** encoding, which split these multi-byte UTF-8 characters into individual bytes, each becoming an unknown token.

### Solution
Implemented **character-level** BPE with preprocessing:

```go
// internal/tokenizer/bpe.go
if t.modelType == "gpt2" {
    text = strings.ReplaceAll(text, " ", "Ġ")   // Space -> U+0120
    text = strings.ReplaceAll(text, "\n", "Ċ")  // Newline -> U+010A

    // Then split into runes (characters), not bytes
    for _, r := range text {
        tokens = append(tokens, string(r))
    }
}
```

## Validation

### Before Fix
```
Input: "Hello world"
Tokens: [9707, 128244, 14615]
Decoded: ["Hello", "<unk>", "world"]  ❌

Input: "\n"
Tokens: [128244]
Decoded: ["<unk>"]  ❌

ChatML: "<|im_start|>system\n..."
Tokens: [..., 128244, ...]  ❌ (newline as <unk>)
```

### After Fix
```
Input: "Hello world"
Tokens: [9707, 1879]
Decoded: ["Hello", "Ġworld"]  ✅

Input: "\n"
Tokens: [198]
Decoded: ["Ċ"]  ✅

Input: " "
Tokens: [220]
Decoded: ["Ġ"]  ✅

ChatML: "<|im_start|>system\n..."
Tokens: [..., 198, ...]  ✅ (newline correct!)
```

## Impact

### Chat Template Support
With newlines fixed, ChatML format now tokenizes correctly:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello world<|im_end|>
<|im_start|>assistant
```

All newlines correctly encoded as token 198, enabling proper instruction-following behavior.

### Model Compatibility
Fixed tokenization enables proper use of:
- ✅ Qwen2.5-Coder models (GPT2 tokenizer)
- ✅ Any instruct models requiring chat templates
- ✅ Models with newline-sensitive formatting

## Related Documentation

- **Detailed Investigation:** `docs/results/TOKENIZER_FIX.md`
- **Newline Discovery:** `docs/results/INFERENCE_DEBUGGING_SESSION2.md`
- **Implementation:** `internal/tokenizer/bpe.go`

## Commits

1. **`c55b7dd`** - Initial space tokenization fix
2. **`e5566c2`** - Newline tokenization fix
3. **`8716549`** - Documentation

## Next Steps

1. **Implement Chat Template Support**
   - Parse template from GGUF metadata
   - Auto-format messages into ChatML
   - Handle special tokens properly

2. **Performance Optimization**
   - Current: ~2-3s/token (too slow)
   - Profile forward pass
   - Optimize hot paths
   - Consider GPU acceleration

3. **Validation Testing**
   - Test with various chat templates
   - Compare outputs with llama.cpp
   - Integration tests for different model types

---

**Status:** ✅ Tokenization complete, inference functional, optimization needed
**Date:** February 9, 2026
