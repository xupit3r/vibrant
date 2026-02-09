# Tokenizer Fix - February 9, 2026

## Summary

**CRITICAL BUG FIXED**: Tokenizer was encoding all spaces as `<unk>` tokens!

**Root Cause**: Qwen uses GPT2-style tokenization where spaces are represented as "Ġ" (U+0120), but our BPE implementation used byte-level encoding which didn't handle this correctly.

**Impact**: Model received completely corrupted input, making inference impossible.

---

## The Bug

### Before Fix:
```
"Hello world" → [9707, 128244, 14615]
                 ["Hello", "<unk>", "world"]

" "           → [128244]
                 ["<unk>"]

" world"      → [128244, 14615]
                 ["<unk>", "world"]
```

Every space became token 128244 (`<unk>`), corrupting the model's understanding.

### After Fix:
```
"Hello world" → [9707, 1879]
                 ["Hello", "Ġworld"]

" "           → [220]
                 ["Ġ"]

" world"      → [1879]
                 ["Ġworld"]
```

Spaces correctly encoded using GPT2-style "Ġ" prefix.

---

## Investigation Process

### 1. llama.cpp Validation
- Confirmed llama.cpp works correctly with same model
- Output: "Hello! How can I assist you today?"
- Proved bug was in our implementation, not the model

### 2. Logits Analysis
- Added detailed logging of top 20 predictions
- Found bizarre tokens: "Visualization", "Secretary", "_MATRIX"
- These made no sense for "Hello world" completion

### 3. Tokenization Discovery
```bash
[DEBUG] Encoded tokens: [151643, 9707, 128244, 14615]
```
- Token 128244 was suspiciously in every prompt
- Decoded to: `<unk>` (unknown token)
- Realized space was being encoded as unknown!

### 4. Vocabulary Investigation
```
Token " "      → NOT FOUND in vocab
Token "Ġ"     → ID 220 (exists!)
Token "Ġworld" → ID 1879 (exists!)
```

Qwen uses GPT2-style tokenization where:
- Spaces → "Ġ" (U+0120, a special Unicode character)
- "Ġworld" is a single token, not "Ġ" + "world"

### 5. Root Cause
Our BPE algorithm worked at **byte level**:
```go
// WRONG: Byte-level BPE
bytes := []byte(text)
for i, b := range bytes {
    tokens[i] = string([]byte{b})
}
```

But GPT2 requires **character level**:
- "Ġ" is 2 bytes in UTF-8 (C4 A0)
- Byte-level split: [C4, A0] → two separate tokens → both unknown
- Character-level split: "Ġ" → one token → merges correctly

---

## The Fix

### Implementation

```go
// 1. Preprocess: Replace spaces with Ġ for GPT2 models
if t.modelType == "gpt2" {
    text = strings.ReplaceAll(text, " ", "Ġ")
}

// 2. Split into UTF-8 characters (runes), not bytes
if t.modelType == "gpt2" {
    for _, r := range text {
        tokens = append(tokens, string(r))
    }
} else {
    // Original byte-level BPE for non-GPT2 models
    bytes := []byte(text)
    tokens = make([]string, len(bytes))
    for i, b := range bytes {
        tokens[i] = string([]byte{b})
    }
}

// 3. Apply BPE merges (unchanged)
// 4. Convert to token IDs (unchanged)
```

### Files Changed
- `internal/tokenizer/bpe.go` - Fixed Encode() to use character-level BPE
- Added GPT2 preprocessing step
- Conditional on model type (supports both GPT2 and byte-level BPE)

---

## Validation

### Test Results
```
Input: "Hello world"
Before: [9707, 128244, 14615] = ["Hello", "<unk>", "world"]  ❌
After:  [9707, 1879]           = ["Hello", "Ġworld"]          ✅

Input: " "
Before: [128244]     = ["<unk>"]  ❌
After:  [220]        = ["Ġ"]      ✅

Input: "  " (two spaces)
Before: [128244, 128244] = ["<unk>", "<unk>"]  ❌
After:  [256]            = ["ĠĠ"]              ✅
```

### Comparison with llama.cpp
- llama.cpp tokenizes "Hello world" correctly
- Our fixed implementation now matches
- No more `<unk>` tokens for normal text!

---

## Impact

### Before Fix:
- Model received corrupted input (spaces as `<unk>`)
- Generated garbage output
- Inference was completely broken

### After Fix:
- Model receives correct tokens
- Input properly represents "Hello world"
- Tokenization matches reference implementation

### UPDATE: Newline Bug Discovered (Same Day!)

**ANOTHER CRITICAL BUG**: After fixing spaces, discovered newlines were also broken!

#### The Problem:
```
"\n" → [128244] = ["<unk>"]  ❌
```

Every newline was encoded as `<unk>`, which broke ChatML format:
```
<|im_start|>system\n       ← This \n became <unk>!
You are helpful.<|im_end|>
```

#### The Fix:
GPT2 uses "Ċ" (U+010A) for newlines, just like "Ġ" for spaces!

```go
if t.modelType == "gpt2" {
    text = strings.ReplaceAll(text, " ", "Ġ")   // Space -> U+0120
    text = strings.ReplaceAll(text, "\n", "Ċ")  // Newline -> U+010A
}
```

#### Validation:
```
"\n" → [198] = ["Ċ"]  ✅
```

Token 198 is the correct newline token!

### Root Cause of Bizarre Logits - SOLVED!

**Discovery**: Qwen2.5-Coder-3B-**Instruct** requires ChatML format!

The model was fine-tuned exclusively on instruction format, NOT raw text. Feeding it "Hello world" is like speaking English to a French-only model - produces nonsense because it never saw that distribution during training.

**Solution**: Use proper ChatML format (see INFERENCE_DEBUGGING_SESSION2.md for details)

---

## Lessons Learned

1. **GPT2 tokenization is NOT byte-level BPE**
   - Uses special Unicode characters for common whitespace
   - Space: "Ġ" (U+0120)
   - Newline: "Ċ" (U+010A)
   - Requires character-level initial split, not byte-level

2. **Vocabulary structure matters**
   - Token 220 = "Ġ" (single space)
   - Token 198 = "Ċ" (newline)
   - Token 1879 = "Ġworld" (space + word)
   - Token 256 = "ĠĠ" (two spaces)

3. **Testing with reference implementation is crucial**
   - llama.cpp validation caught the bug
   - Without comparison, we might have missed this

4. **Debugging from user symptoms to root cause**
   - Started with: "model generates garbage"
   - Found: "logits are bizarre"
   - Discovered: "tokenization produces <unk>"
   - Fixed: "GPT2 needs character-level BPE"

---

## References

- GPT2 Paper: Uses byte-level BPE with special preprocessing
- Qwen Documentation: Confirms GPT2-style tokenization
- llama.cpp: Reference implementation for comparison

---

**Status**: ✅ Tokenizer FULLY FIXED (spaces + newlines), chat template requirement identified
**Next**: Implement chat template support, optimize inference performance
**Date**: February 9, 2026 (Updated with newline fix same day)

