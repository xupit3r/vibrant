# LLM Integration Specification

## Overview
Integration with llama.cpp for CPU-optimized inference of GGUF models.

## Inference Engine

### llama.cpp Bindings
- **Library**: github.com/go-skynet/go-llama.cpp or direct CGO
- **Model Format**: GGUF
- **Quantization**: Q4_K_M, Q5_K_M, Q8_0

### Loading Models
```go
type LLamaLoader struct {
    Options LoadOptions
}

type LoadOptions struct {
    ContextSize int
    Threads     int
    BatchSize   int
    UseMMap     bool
    UseMlock    bool
}
```

### Inference API
```go
type InferenceEngine interface {
    Generate(prompt string, opts GenerateOptions) (string, error)
    GenerateStream(prompt string, opts GenerateOptions) (<-chan string, error)
    TokenCount(text string) int
    Unload() error
}

type GenerateOptions struct {
    MaxTokens   int
    Temperature float32
    TopP        float32
    TopK        int
    StopTokens  []string
}
```

## Status
- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Implementation**: Phase 3
