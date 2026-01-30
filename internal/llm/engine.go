package llm

import (
"context"
"fmt"
"runtime"

llama "github.com/go-skynet/go-llama.cpp"
)

// Engine provides LLM inference capabilities
type Engine interface {
Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error)
GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error)
TokenCount(text string) int
Close() error
}

// GenerateOptions configures text generation
type GenerateOptions struct {
MaxTokens   int     // Maximum tokens to generate
Temperature float32 // Randomness (0.0-1.0)
TopP        float32 // Nucleus sampling
TopK        int     // Top-k sampling
StopTokens  []string // Stop sequences
}

// LoadOptions configures model loading
type LoadOptions struct {
ContextSize int  // Max context tokens
Threads     int  // CPU threads for inference
BatchSize   int  // Batch size for prompt processing
UseMMap     bool // Use memory-mapped files
UseMlock    bool // Lock pages in RAM
Verbose     bool // Enable verbose logging
}

// DefaultLoadOptions returns sensible defaults
func DefaultLoadOptions() LoadOptions {
return LoadOptions{
ContextSize: 4096,
Threads:     runtime.NumCPU(),
BatchSize:   512,
UseMMap:     true,
UseMlock:    false,
Verbose:     false,
}
}

// DefaultGenerateOptions returns sensible defaults for code generation
func DefaultGenerateOptions() GenerateOptions {
return GenerateOptions{
MaxTokens:   1024,
Temperature: 0.2, // Low temperature for deterministic code
TopP:        0.95,
TopK:        40,
StopTokens:  []string{},
}
}

// LlamaEngine wraps go-llama.cpp
type LlamaEngine struct {
model   *llama.LLama
options LoadOptions
}

// NewLlamaEngine creates a new LLama inference engine
func NewLlamaEngine(modelPath string, opts LoadOptions) (*LlamaEngine, error) {
// Initialize llama.cpp model
model, err := llama.New(
modelPath,
llama.SetContext(opts.ContextSize),
llama.SetParts(-1),
llama.SetThreads(opts.Threads),
llama.EnableF16Memory,
llama.EnableEmbeddings,
)
if err != nil {
return nil, fmt.Errorf("failed to load model: %w", err)
}

return &LlamaEngine{
model:   model,
options: opts,
}, nil
}

// Generate generates text from a prompt (blocking)
func (e *LlamaEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
result := ""

// Use Predict with callback to accumulate text
_, err := e.model.Predict(
prompt,
llama.SetTokens(opts.MaxTokens),
llama.SetTemperature(float64(opts.Temperature)),
llama.SetTopP(float64(opts.TopP)),
llama.SetTopK(opts.TopK),
llama.SetTokenCallback(func(token string) bool {
result += token

// Check context cancellation
select {
case <-ctx.Done():
return false // Stop generation
default:
return true // Continue
}
}),
)

if err != nil {
return "", fmt.Errorf("generation failed: %w", err)
}

return result, nil
}

// GenerateStream generates text from a prompt (streaming)
func (e *LlamaEngine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
ch := make(chan string, 10)

go func() {
defer close(ch)

_, err := e.model.Predict(
prompt,
llama.SetTokens(opts.MaxTokens),
llama.SetTemperature(float64(opts.Temperature)),
llama.SetTopP(float64(opts.TopP)),
llama.SetTopK(opts.TopK),
llama.SetTokenCallback(func(token string) bool {
// Check context cancellation
select {
case <-ctx.Done():
return false
case ch <- token:
return true
}
}),
)

if err != nil {
// Send error as special token (not ideal but works for now)
select {
case ch <- fmt.Sprintf("\nError: %v", err):
case <-ctx.Done():
}
}
}()

return ch, nil
}

// TokenCount estimates token count for text
func (e *LlamaEngine) TokenCount(text string) int {
// Simple estimation: ~4 chars per token for code
// This is a rough approximation
return len(text) / 4
}

// Close releases model resources
func (e *LlamaEngine) Close() error {
if e.model != nil {
e.model.Free()
e.model = nil
}
return nil
}
