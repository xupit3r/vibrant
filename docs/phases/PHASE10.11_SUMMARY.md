# Phase 10.11: Chat Templates, Cache Warming & Fused Dequant-Transpose

## Date: 2026-02-10

## Overview

Phase 10.11 addresses four issues blocking useful output from Vibrant's custom inference engine:

1. **No chat template** - Instruct models (Qwen2.5-Coder-3B-Instruct) require specific prompt formatting (ChatML, Llama3); raw text produces nonsense.
2. **Slow first inference** - All 253 quantized weight matrices were dequantized+transposed on first use, making the first forward pass extremely slow.
3. **Memory pressure** - Separate dequant + transpose required two large transient allocations per weight.
4. **Debug spam** - `model.go` printed `[MODEL] ...` on every forward pass unconditionally.

## Changes

### 1. Debug Output Cleanup

**File**: `internal/transformer/model.go`

- Gated all `[MODEL]` output projection prints behind `debugForward` flag (default `false`)
- Commented out one-time config print in `NewModel()`
- Removed `time` import (no longer needed)

### 2. Chat Template Support

**New package**: `internal/chat/`

Multi-format auto-detect chat template system:

| Format | Detection | Stop Token | Models |
|--------|-----------|------------|--------|
| ChatML | `<\|im_start\|>` in GGUF template | `<\|im_end\|>` | Qwen, Yi |
| Llama 3 | `<\|start_header_id\|>` in GGUF template | `<\|eot_id\|>` | Llama 3/3.1 |
| Plain | Fallback | EOS only | Base models |

**Files created/modified**:
- `internal/chat/template.go` - ChatTemplate type with Format/FormatSimple methods
- `internal/chat/template_test.go` - 10 tests covering all formats
- `internal/gguf/metadata.go` - Added `KeyChatTemplate` constant
- `internal/gguf/helpers.go` - Added `GetChatTemplate()` method
- `internal/inference/engine.go` - Auto-detect template, register stop tokens, store template
- `internal/llm/engine_custom.go` - Added `FormatPrompt()` method
- `internal/llm/manager.go` - Added `FormatPrompt()` delegation
- `cmd/vibrant/commands/ask.go` - Uses `llmMgr.FormatPrompt()` for prompt building

**Special token encoding**:
- `internal/tokenizer/bpe.go` - New `encodeWithSpecialTokenSplit()` method
  - Lazily builds sorted list of `<|...|>` tokens from vocab
  - Splits text at special token boundaries
  - Regular text goes through BPE, special tokens looked up directly
- `internal/tokenizer/tokenizer_test.go` - 5 new tests for special token encoding

### 3. Cache Warming

**File**: `internal/transformer/model.go`

Added `WarmWeightCache()` method that pre-dequantizes and transposes all quantized weight matrices at model load time:
- Iterates all layers: `attn.{wq,wk,wv,wo}`, `ffn.{gate,up,down}`
- Also warms the output weight cache
- Called from `NewEngine()` after model creation

**Expected cost**: ~5-13s for Qwen2.5-3B (253 weights). One-time at load.

### 4. Fused Dequant-Transpose

**New file**: `internal/tensor/dequant_transpose.go`

Three fused functions (`DequantTransposeQ4K/Q5K/Q6K`) that:
1. Allocate a single `[N, M]` float32 result
2. Dequantize each block into a small local buffer (1KB)
3. Scatter-write to transposed positions

**Memory savings**: 50% reduction in peak memory per weight (one allocation vs two).

**Integration**: `GetOrDequantTranspose()` uses fused path for 2D Q4_K/Q5_K/Q6_K tensors, falls back to separate path otherwise.

**Tests**: `internal/tensor/dequant_transpose_test.go`
- Correctness: verifies fused output matches separate dequant+transpose
- Benchmarks: fused vs separate for Q4K, Q5K, Q6K at 768x768

## Test Results

All tests pass:
```
ok  github.com/xupit3r/vibrant/internal/assistant     0.004s
ok  github.com/xupit3r/vibrant/internal/chat           0.002s
ok  github.com/xupit3r/vibrant/internal/gguf           0.415s
ok  github.com/xupit3r/vibrant/internal/inference      0.003s
ok  github.com/xupit3r/vibrant/internal/llm            3.410s
ok  github.com/xupit3r/vibrant/internal/tensor         0.232s
ok  github.com/xupit3r/vibrant/internal/tokenizer      0.003s
ok  github.com/xupit3r/vibrant/internal/transformer    0.042s
ok  github.com/xupit3r/vibrant/cmd/vibrant/commands    0.014s
```

## Architecture Decision: `internal/chat/` Package

The chat template was initially placed in `internal/assistant/` but this created an import cycle:
```
inference → assistant → llm → inference
```

Solution: Created standalone `internal/chat/` package with zero internal dependencies.
Both `inference` and `assistant` can import it without cycles.

## Files Summary

| Category | File | Action |
|----------|------|--------|
| Chat templates | `internal/chat/template.go` | Created |
| Chat templates | `internal/chat/template_test.go` | Created |
| Fused dequant | `internal/tensor/dequant_transpose.go` | Created |
| Fused dequant | `internal/tensor/dequant_transpose_test.go` | Created |
| GGUF metadata | `internal/gguf/metadata.go` | Modified |
| GGUF metadata | `internal/gguf/helpers.go` | Modified |
| Inference | `internal/inference/engine.go` | Modified |
| LLM layer | `internal/llm/engine_custom.go` | Modified |
| LLM layer | `internal/llm/manager.go` | Modified |
| CLI | `cmd/vibrant/commands/ask.go` | Modified |
| Tokenizer | `internal/tokenizer/bpe.go` | Modified |
| Tokenizer | `internal/tokenizer/tokenizer_test.go` | Modified |
| Transformer | `internal/transformer/model.go` | Modified |
