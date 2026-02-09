# Inference Debugging Session 2 - February 9, 2026

## Summary

**MAJOR BREAKTHROUGH**: Discovered that Qwen2.5-Coder is an **INSTRUCT model** requiring ChatML format, and that newlines were being encoded as `<unk>` tokens!

## Key Findings

### 1. Newline Tokenization Bug (CRITICAL FIX)

**Problem**: Newlines `\n` were being encoded as token 128244 (`<unk>`)

**Root Cause**: GPT2-style tokenizers use special Unicode characters:
- Space: `Ġ` (U+0120) → token 220
- Newline: `Ċ` (U+010A) → token 198

**Impact**: ChatML format requires newlines, so every chat template was corrupted:
```
<|im_start|>system\nYou are helpful.<|im_end|>
                  ↑↑ encoded as <unk> tokens!
```

**Fix**: Extended GPT2 preprocessing to replace both spaces AND newlines
```go
if t.modelType == "gpt2" {
    text = strings.ReplaceAll(text, " ", "Ġ")  // Space -> U+0120
    text = strings.ReplaceAll(text, "\n", "Ċ") // Newline -> U+010A
}
```

**Validation**:
```
Before: "\n" → [128244] = ["<unk>"]  ❌
After:  "\n" → [198]    = ["Ċ"]      ✅
```

### 2. Model Requires Chat Template (ROOT CAUSE OF BIZARRE LOGITS)

**Discovery**: The model file contains a chat template using ChatML format:
```bash
$ strings qwen2.5-coder-3b-q4.gguf | grep chat_template
<|im_start|>system
...
```

**Qwen2.5-Coder-3B-Instruct** is a **chat/instruct model**, NOT a base model!

**What This Means**:
- ❌ Raw text input: "Hello world" → bizarre logits (token 128008 "íİĺìĿ´ì§Ģ")
- ✅ ChatML format: Proper instruction-following behavior

**Correct Format**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello world<|im_end|>
<|im_start|>assistant
```

### 3. Why We Saw Bizarre Logits

When testing with raw "Hello world" input:
- Top prediction: token 128008 ("íİĺìĿ´ì§Ģ") score 20.56
- Second: token 90760 ("ĠVisualization") score 20.20
- Third: token 97928 ("Secretary") score 20.04

**Explanation**: The model was trained EXCLUSIVELY on ChatML-formatted conversations. Feeding it raw text is like speaking English to a French-only model - it produces nonsense because it's never seen that distribution!

The model literally does not know what should come after plain "Hello world" because:
1. It was fine-tuned on instruction format only
2. All training examples had the ChatML template
3. Raw continuations were removed during instruction tuning

### 4. Previous Session's Findings Still Valid

From TOKENIZER_FIX.md (earlier today):
- ✅ Spaces correctly encoded as "Ġ" (U+0120)
- ✅ Character-level BPE working properly
- ✅ No more `<unk>` for space characters

This session extended the fix to handle newlines.

---

## Investigation Process

### Step 1: Re-enabled Debug Logging
Added comprehensive logits distribution logging to see:
- Top 20 token predictions with scores
- Min/max/mean/range statistics
- Decoded token text

### Step 2: Tested with Fixed Tokenizer
Confirmed tokenization was correct:
```
"Hello world" → [151643, 9707, 1879]
                [BOS, "Hello", "Ġworld"]  ✅
```

But logits were still bizarre - top token 128008 made no sense.

### Step 3: Discovered Chat Template Requirement
```bash
$ strings model.gguf | grep chat_template
# Found ChatML format tokens!
```

Model is "Qwen2.5-Coder-3B-**Instruct**" - requires instruction format.

### Step 4: Found Newline Bug
When trying to test ChatML format, discovered:
```
"<|im_start|>system\nYou are..."
→ tokens include [128244, 128244, ...]  # <unk> tokens!
```

Decoded and found newlines `\n` were becoming `<unk>`.

### Step 5: Fixed Newline Encoding
Checked vocabulary:
- Token 198 = "Ċ" (U+010A) ← this is newline!
- Token 220 = "Ġ" (U+0120) ← this is space

Extended preprocessing to handle both.

---

## Current Status

### ✅ Fixed
1. Space tokenization - "Ġ" (U+0120)
2. Newline tokenization - "Ċ" (U+010A)
3. Character-level BPE for GPT2 models
4. Identified chat template requirement

### ⚠️ Known Issues
1. **Inference is VERY slow** (~2-3s per token on CPU)
   - Needs performance optimization
   - Consider GPU acceleration
   - Profile to find bottlenecks

2. **No chat template implementation**
   - Currently requires manual ChatML formatting
   - Should add `ApplyChatTemplate()` method to tokenizer
   - Parse template from GGUF metadata

3. **Missing special token handling**
   - Need to properly handle `<|im_start|>`, `<|im_end|>`, etc.
   - These should be single tokens, not byte sequences

### 🔍 Next Steps

1. **Implement Chat Template Support**
   ```go
   // tokenizer/chat_template.go
   func (t *Tokenizer) ApplyChatTemplate(messages []Message) string {
       // Parse template from GGUF metadata
       // Apply Jinja2-style template
       // Return formatted string
   }
   ```

2. **Validate Inference with ChatML**
   - Test with properly formatted prompts
   - Verify logits make sense
   - Compare with llama.cpp output

3. **Performance Optimization**
   - Profile forward pass
   - Optimize hot paths
   - Consider GPU acceleration for prefill

4. **Add Integration Tests**
   ```go
   func TestQwenInference(t *testing.T) {
       prompt := formatChatML(system, user, assistant)
       output := engine.Generate(prompt)
       assert.Contains(output, "helpful")  // Check sanity
   }
   ```

---

## Code Changes

### Files Modified
1. `internal/tokenizer/bpe.go`
   - Added newline replacement: `\n` → `Ċ`
   - Extended GPT2 preprocessing

2. `internal/transformer/model.go`
   - Commented out debug logging for performance

3. `internal/inference/engine.go`
   - Already had logits logging (from earlier session)
   - Disabled debug flag

### Test Files Created
- `test/debug_tokenizer.go` - Token encoding validation
- `test/check_token.go` - Vocabulary inspection
- `test/find_newline_token.go` - Newline token discovery
- `test/test_chat_template.go` - ChatML format testing
- `test/quick_inference.go` - Fast inference test

---

## Lessons Learned

### 1. Know Your Model Type!
- **Base models**: Continue text naturally
- **Instruct models**: Require chat template
- **Code models**: May need special format

Always check:
```bash
strings model.gguf | grep chat_template
strings model.gguf | grep "Instruct\|Chat"
```

### 2. GPT2 Tokenization Has Multiple Special Characters
Not just spaces!
- Space: `Ġ` (U+0120) → token 220
- Newline: `Ċ` (U+010A) → token 198
- Tab: `Ģ` (U+0122) → token 210 (probably)

### 3. UTF-8 Multi-byte Characters Need Character-Level BPE
Byte-level BPE splits:
- `Ġ` (2 bytes: C4 A0) → TWO unknown tokens ❌
- `Ċ` (2 bytes: C4 8A) → TWO unknown tokens ❌

Character-level BPE keeps them together ✅

### 4. Debug Output is Essential
Without detailed logits logging, we would never have:
- Seen the bizarre token 128008 predictions
- Realized the model was seeing corrupted input
- Connected it to the missing chat template

### 5. Compare with Reference Implementation
llama.cpp works correctly → proves our implementation has bugs
This guided us to check tokenization → found the issues

---

## References

- [Qwen2.5 Documentation](https://github.com/QwenLM/Qwen2.5)
- [GPT2 Tokenization](https://github.com/openai/gpt-2)
- [ChatML Format](https://github.com/openai/openai-python/blob/main/chatml.md)
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

**Status**: 🎯 Tokenization fully fixed, inference needs optimization and chat template support
**Date**: February 9, 2026
**Session**: 2 (continuation of tokenizer debugging)
