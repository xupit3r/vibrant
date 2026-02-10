# LLM Integration Specification

## Overview
Integration with llama.cpp for CPU-optimized inference of GGUF models. **Llama.cpp is enabled by default** in production builds, with a mock engine available for testing/development.

## Build Configuration

### Default Build (llama.cpp enabled)
```bash
make build
# or manually:
CGO_ENABLED=1 go build -tags llama -o vibrant ./cmd/vibrant
```

### Mock Engine Build (for testing)
```bash
make build-mock
# or manually:
go build -o vibrant ./cmd/vibrant
```

### Build Tags
- `+build llama` - Enables real llama.cpp inference (in `engine.go`)
- `+build !llama` - Enables mock engine (in `engine_mock.go`)

## Inference Engine

### llama.cpp Bindings
- **Library**: github.com/go-skynet/go-llama.cpp or direct CGO
- **Model Format**: GGUF
- **Quantization**: Q4_K_M, Q5_K_M, Q8_0
- **Requirements**: C++ compiler (gcc/clang), CGO_ENABLED=1

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

## Custom Pure Go Engine (Phase 10+)

The default engine is now a pure Go inference engine (`internal/inference/`). No CGO or external
dependencies required.

### Chat Template Support (Phase 10.11)

The engine auto-detects the model's chat template from GGUF metadata and applies it during
prompt formatting. Supported formats: ChatML, Llama 3, plain text fallback.

```go
// FormatPrompt applies the model's chat template
prompt := llmMgr.FormatPrompt(systemPrompt, userQuestion)

// Or directly via the engine
prompt := engine.ChatTemplate().FormatSimple(system, user)
```

See `specs/custom-inference.md` for full details.

## Status
- **Current**: Pure Go engine with chat templates, cache warming, fused dequant
- **Last Updated**: 2026-02-10
- **Implementation**: Phase 10.11 (complete)
