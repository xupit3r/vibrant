# Custom Inference Engine Specification

## Overview

The custom inference engine (`internal/inference/`) implements a pure Go LLM inference pipeline for transformer models, specifically optimized for Qwen 2.5 Coder models.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              CustomEngine                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │  Model   │  │ Tokenizer │  │   Sampler    │ │
│  │(Qwen2.5) │  │   (BPE)   │  │ (Top-p/Top-k)│ │
│  └─────┬────┘  └─────┬─────┘  └──────┬───────┘ │
│        │             │                │         │
│  ┌─────▼─────────────▼────────────────▼──────┐  │
│  │          Inference Pipeline               │  │
│  │  ┌─────────┐  ┌───────┐  ┌────────────┐  │  │
│  │  │ Prefill │─▶│ Decode│─▶│   Stream   │  │  │
│  │  └─────────┘  └───────┘  └────────────┘  │  │
│  └──────────────────────────────────────────┘  │
│                      │                          │
│            ┌─────────▼───────────┐              │
│            │     KV Cache        │              │
│            │  (Per-layer cache)  │              │
│            └─────────────────────┘              │
└─────────────────────────────────────────────────┘
```

## Core Components

### 1. Inference Engine

```go
type CustomEngine struct {
    model     *transformer.Model
    tokenizer *tokenizer.BPETokenizer
    cache     *KVCache
    sampler   *Sampler
    config    *Config
}

type Config struct {
    MaxTokens    int
    Temperature  float32
    TopP         float32
    TopK         int
    StopTokens   []string
    Stream       bool
}

// Implements the llm.Engine interface
func (e *CustomEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error)
func (e *CustomEngine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error)
func (e *CustomEngine) TokenCount(text string) int
func (e *CustomEngine) Close() error
```

### 2. KV-Cache Management

```go
type KVCache struct {
    numLayers int
    keys      [][]*tensor.Tensor  // [layer][batch, num_heads, seq_len, head_dim]
    values    [][]*tensor.Tensor  // [layer][batch, num_heads, seq_len, head_dim]
    seqLen    int                 // Current sequence length
    maxSeqLen int                 // Maximum supported sequence length
}

func NewKVCache(numLayers, maxSeqLen, numHeads, headDim int) *KVCache
func (c *KVCache) Update(layer int, k, v *tensor.Tensor)
func (c *KVCache) Get(layer int) (k, v *tensor.Tensor)
func (c *KVCache) Clear()
func (c *KVCache) Resize(newMaxSeqLen int)
```

### 3. Sampling Strategies

```go
type Sampler struct {
    temperature float32
    topP        float32
    topK        int
    rng         *rand.Rand
}

// Sample selects next token from logits
func (s *Sampler) Sample(logits *tensor.Tensor) int {
    // 1. Apply temperature scaling
    // 2. Top-K filtering (keep only top K tokens)
    // 3. Top-P (nucleus) filtering (keep tokens with cumulative prob < P)
    // 4. Sample from filtered distribution
}

// Greedy sampling (for deterministic output)
func (s *Sampler) SampleGreedy(logits *tensor.Tensor) int
```

### 4. Inference Pipeline

```go
// Two-stage inference: Prefill → Decode
type Pipeline struct {
    engine *CustomEngine
}

// Prefill: Process all prompt tokens in parallel
func (p *Pipeline) Prefill(ctx context.Context, tokens []int) (*tensor.Tensor, error) {
    // Forward pass through model with all prompt tokens
    // Returns logits for last token
}

// Decode: Generate tokens one at a time
func (p *Pipeline) Decode(ctx context.Context, token int) (*tensor.Tensor, error) {
    // Forward pass with single token
    // Uses cached K/V from previous steps
}
```

## Inference Flow

### Generate (Blocking)

```go
func (e *CustomEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
    // 1. Tokenize prompt
    tokens := e.tokenizer.Encode(prompt)

    // 2. Prefill: process all prompt tokens
    logits, err := e.model.Forward(ctx, tokens, e.cache)
    if err != nil {
        return "", err
    }

    // 3. Decode: generate tokens one by one
    generatedTokens := []int{}
    for i := 0; i < opts.MaxTokens; i++ {
        // Check context cancellation
        select {
        case <-ctx.Done():
            return "", ctx.Err()
        default:
        }

        // Sample next token
        nextToken := e.sampler.Sample(logits)
        generatedTokens = append(generatedTokens, nextToken)

        // Check for stop tokens
        if e.isStopToken(nextToken, opts.StopTokens) {
            break
        }

        // Decode: forward pass with single token
        logits, err = e.model.Forward(ctx, []int{nextToken}, e.cache)
        if err != nil {
            return "", err
        }
    }

    // 4. Decode tokens to text
    return e.tokenizer.Decode(generatedTokens), nil
}
```

### GenerateStream (Non-blocking)

```go
func (e *CustomEngine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
    ch := make(chan string, 16) // Buffered for throughput

    go func() {
        defer close(ch)

        // Tokenize prompt
        tokens := e.tokenizer.Encode(prompt)

        // Prefill stage
        logits, err := e.model.Forward(ctx, tokens, e.cache)
        if err != nil {
            ch <- fmt.Sprintf("ERROR: %v", err)
            return
        }

        // Decode stage: stream tokens as generated
        for i := 0; i < opts.MaxTokens; i++ {
            select {
            case <-ctx.Done():
                return
            default:
            }

            // Sample next token
            nextToken := e.sampler.Sample(logits)

            // Decode token to text and send immediately
            tokenText := e.tokenizer.Decode([]int{nextToken})
            ch <- tokenText

            // Check stop conditions
            if e.isStopToken(nextToken, opts.StopTokens) {
                break
            }

            // Continue decoding
            logits, err = e.model.Forward(ctx, []int{nextToken}, e.cache)
            if err != nil {
                ch <- fmt.Sprintf("ERROR: %v", err)
                return
            }
        }
    }()

    return ch, nil
}
```

## Sampling Algorithms

### Temperature Scaling

```go
func applyTemperature(logits *tensor.Tensor, temp float32) {
    if temp == 0 {
        // Greedy: set max to large value, rest to zero
        maxIdx := tensor.ArgMax(logits)
        logits.SetAll(float32(math.Inf(-1)))
        logits.Set(maxIdx, 1e10)
        return
    }

    // Scale logits: logits /= temperature
    tensor.DivScalar(logits, temp)
}
```

### Top-K Filtering

```go
func applyTopK(logits *tensor.Tensor, k int) {
    if k <= 0 || k >= logits.Size() {
        return // No filtering
    }

    // Find k-th largest value
    sorted := tensor.Sort(logits, descending=true)
    threshold := sorted.At(k)

    // Mask out values below threshold
    for i := 0; i < logits.Size(); i++ {
        if logits.At(i) < threshold {
            logits.Set(i, float32(math.Inf(-1)))
        }
    }
}
```

### Top-P (Nucleus) Filtering

```go
func applyTopP(logits *tensor.Tensor, p float32) {
    if p >= 1.0 {
        return // No filtering
    }

    // Convert to probabilities
    probs := tensor.Softmax(logits, dim=0)

    // Sort in descending order
    sorted, indices := tensor.SortWithIndices(probs, descending=true)

    // Compute cumulative sum
    cumsum := make([]float32, len(sorted))
    cumsum[0] = sorted[0]
    for i := 1; i < len(sorted); i++ {
        cumsum[i] = cumsum[i-1] + sorted[i]
    }

    // Find cutoff index where cumsum exceeds p
    cutoffIdx := 0
    for i, cs := range cumsum {
        if cs >= p {
            cutoffIdx = i
            break
        }
    }

    // Mask out tokens after cutoff
    for i := cutoffIdx + 1; i < len(indices); i++ {
        logits.Set(indices[i], float32(math.Inf(-1)))
    }
}
```

### Multinomial Sampling

```go
func sampleMultinomial(logits *tensor.Tensor, rng *rand.Rand) int {
    // Convert logits to probabilities
    probs := tensor.Softmax(logits, dim=0)

    // Sample from categorical distribution
    r := rng.Float32()
    cumsum := float32(0)
    for i := 0; i < probs.Size(); i++ {
        cumsum += probs.At(i)
        if r <= cumsum {
            return i
        }
    }

    // Fallback (shouldn't happen)
    return probs.Size() - 1
}
```

## Chat Template Support

The inference engine auto-detects and applies model-specific chat templates.

### Architecture

```
┌─────────────────────────────────────────────┐
│              NewEngine()                     │
│                                              │
│  1. Parse GGUF                               │
│  2. Load model + tokenizer                   │
│  3. Detect chat template from GGUF metadata  │
│  4. Register stop tokens per template        │
│  5. Warm weight cache                        │
└─────────────────────────────────────────────┘
```

### Supported Formats

| Format | Models | Detection | Stop Token |
|--------|--------|-----------|------------|
| ChatML | Qwen, Yi | `<\|im_start\|>` in template | `<\|im_end\|>` |
| Llama 3 | Llama 3/3.1 | `<\|start_header_id\|>` in template | `<\|eot_id\|>` |
| Plain text | Base models | Fallback | EOS only |

### Chat Template Package (`internal/chat/`)

```go
// Auto-detect from raw GGUF template string
ct := chat.NewChatTemplate(rawTemplate)

// Format messages
prompt := ct.FormatSimple("You are helpful.", "Write hello world in Go")

// Or multi-turn
prompt := ct.Format([]chat.Message{
    {Role: "system", Content: "You are helpful."},
    {Role: "user", Content: "What is 2+2?"},
    {Role: "assistant", Content: "4"},
    {Role: "user", Content: "And 3+3?"},
})
```

### Integration Flow

1. **Engine initialization**: `NewEngine()` reads `tokenizer.chat_template` from GGUF
2. **Stop tokens**: Template stop token + EOS token added to config
3. **Prompt formatting**: `FormatPrompt()` on CustomEngine/Manager applies template
4. **Ask command**: `buildPromptWithContext()` calls `llmMgr.FormatPrompt()`

## Weight Cache Warming

At model load time, `WarmWeightCache()` pre-dequantizes and transposes all quantized
weight matrices. This eliminates the cold-cache penalty on the first forward pass.

```go
// Called automatically in NewEngine() after model creation
model.WarmWeightCache()
```

- Iterates all layers, calling `GetOrDequantTranspose()` on each weight tensor
- Also dequantizes the output weight matrix
- One-time cost at load (~5-13s for Qwen2.5-3B)

## Fused Dequant-Transpose

`GetOrDequantTranspose()` uses fused functions that dequantize directly into transposed
layout, reducing peak memory by ~50% per weight (one allocation instead of two):

```go
// Standard: dequant (alloc N*M) → transpose (alloc N*M) → discard first
// Fused:    alloc N*M → dequant blocks → scatter-write to transposed positions
```

Supported: `DequantTransposeQ4K`, `DequantTransposeQ5K`, `DequantTransposeQ6K`

Falls back to separate dequant+transpose for non-2D tensors or unsupported types.

## Performance Optimizations

### 1. Prefill vs Decode Optimization

**Prefill Stage** (compute-bound):
- Process many tokens in parallel
- Large matrix multiplications (GEMM)
- Can use batching and parallelism

**Decode Stage** (memory-bound):
- Process one token at a time
- Matrix-vector multiplications (GEMV)
- Bottleneck: KV-cache memory bandwidth

### 2. KV-Cache Layout

```go
// Optimal memory layout for cache-friendly access
type KVCache struct {
    // Contiguous memory per layer
    keysData   [][]float32  // [layer][seq_len * num_heads * head_dim]
    valuesData [][]float32

    // Metadata for reshaping
    numHeads   int
    headDim    int
    seqLen     int
}

// Access pattern optimized for sequential generation
func (c *KVCache) GetKeySlice(layer, head int) []float32 {
    start := head * c.headDim * c.seqLen
    end := start + c.headDim * c.seqLen
    return c.keysData[layer][start:end]
}
```

### 3. Batch Processing (Future)

```go
// Support for processing multiple prompts simultaneously
type BatchedEngine struct {
    engine     *CustomEngine
    maxBatchSize int
}

func (b *BatchedEngine) GenerateBatch(ctx context.Context, prompts []string, opts GenerateOptions) ([]string, error) {
    // Tokenize all prompts
    // Pad to same length
    // Single forward pass for entire batch
    // Decode all in parallel
}
```

## Integration with LLM Interface

```go
// internal/llm/engine_custom.go

// +build !llama

package llm

import (
    "vibrant/internal/inference"
    "vibrant/internal/gguf"
    "vibrant/internal/transformer"
    "vibrant/internal/tokenizer"
)

// NewEngine creates a custom inference engine
func NewEngine(modelPath string, opts LoadOptions) (Engine, error) {
    // Parse GGUF file
    ggufFile, err := gguf.ParseGGUF(modelPath)
    if err != nil {
        return nil, fmt.Errorf("failed to parse GGUF: %w", err)
    }

    // Build transformer model
    model, err := transformer.NewModelFromGGUF(ggufFile)
    if err != nil {
        return nil, fmt.Errorf("failed to build model: %w", err)
    }

    // Create tokenizer
    tok, err := tokenizer.NewBPETokenizer(ggufFile)
    if err != nil {
        return nil, fmt.Errorf("failed to create tokenizer: %w", err)
    }

    // Create KV-cache
    cache := inference.NewKVCache(
        model.NumLayers(),
        opts.ContextSize,
        model.NumHeads(),
        model.HeadDim(),
    )

    // Create sampler
    sampler := inference.NewSampler(opts.Temperature, opts.TopP, opts.TopK)

    // Create engine
    engine := &inference.CustomEngine{
        Model:     model,
        Tokenizer: tok,
        Cache:     cache,
        Sampler:   sampler,
    }

    return engine, nil
}
```

## Testing Strategy

### Unit Tests

```go
func TestKVCache(t *testing.T) {
    cache := NewKVCache(numLayers=24, maxSeqLen=4096, numHeads=28, headDim=128)

    // Test update
    k := tensor.NewTensor([]int{1, 28, 1, 128}, tensor.Float32)
    v := tensor.NewTensor([]int{1, 28, 1, 128}, tensor.Float32)
    cache.Update(0, k, v)

    // Test retrieval
    kCached, vCached := cache.Get(0)
    if kCached.Shape()[2] != 1 {
        t.Errorf("Expected seq_len=1, got %d", kCached.Shape()[2])
    }
}

func TestSampler(t *testing.T) {
    logits := tensor.NewTensorFromData([]float32{1.0, 2.0, 3.0}, []int{3})

    sampler := NewSampler(temp=1.0, topP=0.9, topK=2)
    token := sampler.Sample(logits)

    // Token should be 1 or 2 (top-2)
    if token != 1 && token != 2 {
        t.Errorf("Unexpected token: %d", token)
    }
}
```

### Integration Tests

**Phase 10.10 Improvements**:
- ✅ Fixed integration test infrastructure
- ✅ Added `TestMain()` to auto-build vibrant binary before tests
- ✅ All 11 integration tests now passing automatically
- ✅ No manual binary build required

```go
// test/integration/completion_test.go
func TestMain(m *testing.M) {
    // Build binary before running tests
    projectRoot, _ := filepath.Abs("../..")
    cmd := exec.Command("go", "build", "-o", "vibrant", "./cmd/vibrant")
    cmd.Dir = projectRoot
    if err := cmd.Run(); err != nil {
        fmt.Fprintf(os.Stderr, "Failed to build vibrant: %v\n", err)
        os.Exit(1)
    }

    // Run tests
    code := m.Run()

    // Cleanup
    os.Remove(filepath.Join(projectRoot, "vibrant"))
    os.Exit(code)
}
```

### End-to-End Tests

```go
func TestEndToEndGeneration(t *testing.T) {
    // Load small test model
    engine, err := NewEngine("testdata/tiny-qwen-q4_k.gguf", LoadOptions{
        ContextSize: 2048,
        Threads:     4,
        Temperature: 0.7,
    })
    if err != nil {
        t.Fatal(err)
    }
    defer engine.Close()

    // Generate response
    response, err := engine.Generate(
        context.Background(),
        "What is Golang?",
        GenerateOptions{MaxTokens: 50},
    )

    if err != nil {
        t.Fatal(err)
    }

    // Verify response is non-empty
    if len(response) == 0 {
        t.Error("Empty response")
    }

    t.Logf("Response: %s", response)
}
```

### Numerical Validation

```go
func TestNumericalAccuracy(t *testing.T) {
    // Compare against llama.cpp outputs
    // 1. Load same model with both engines
    // 2. Feed identical input
    // 3. Compare logits (should be within 1e-4)

    customEngine := loadCustomEngine("model.gguf")
    llamaEngine := loadLlamaCppEngine("model.gguf")

    prompt := "func main() {"
    tokens := tokenize(prompt)

    customLogits := customEngine.Forward(tokens)
    llamaLogits := llamaEngine.Forward(tokens)

    // Compare all logits
    for i := range customLogits {
        diff := math.Abs(customLogits[i] - llamaLogits[i])
        if diff > 1e-4 {
            t.Errorf("Logit mismatch at %d: diff=%e", i, diff)
        }
    }
}
```

## Performance Benchmarks

```go
func BenchmarkPrefill(b *testing.B) {
    engine := setupTestEngine()
    tokens := makeTokens(512) // 512-token prompt

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        engine.model.Forward(context.Background(), tokens, engine.cache)
    }
}

func BenchmarkDecode(b *testing.B) {
    engine := setupTestEngine()

    // Prefill first
    engine.model.Forward(context.Background(), makeTokens(128), engine.cache)

    // Benchmark decode
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        engine.model.Forward(context.Background(), []int{1234}, engine.cache)
    }
}

func BenchmarkEndToEnd(b *testing.B) {
    engine := setupTestEngine()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        engine.Generate(
            context.Background(),
            "Write a hello world program in Go",
            GenerateOptions{MaxTokens: 100},
        )
    }
}
```

## Future Enhancements

1. **Speculative Decoding**: Use small model to draft, large model to verify
2. **Continuous Batching**: Dynamic batch size based on load
3. **Quantized KV-Cache**: Reduce memory usage with Q8/Q4 cache
4. **Multi-GPU Support**: Tensor parallelism for large models
5. **Flash Attention**: Memory-efficient attention implementation
6. **Model Parallelism**: Split layers across multiple devices

## References

- [llama.cpp Inference Pipeline](https://github.com/ggml-org/llama.cpp)
- [Hugging Face Transformers Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
