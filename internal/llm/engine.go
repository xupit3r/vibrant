// +build llama

package llm

import (
"context"
"fmt"

llama "github.com/go-skynet/go-llama.cpp"
)

// NewLlamaEngine creates a new LLama inference engine
func NewLlamaEngine(modelPath string, opts LoadOptions) (*LlamaEngine, error) {
// Initialize llama.cpp model
llamaModel, err := llama.New(
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
model:   llamaModel,
options: opts,
}, nil
}

// Generate generates text from a prompt (blocking)
func (e *LlamaEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
result := ""
llamaModel := e.model.(*llama.LLama)

// Use Predict with callback to accumulate text
_, err := llamaModel.Predict(
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
llamaModel := e.model.(*llama.LLama)

go func() {
defer close(ch)

_, err := llamaModel.Predict(
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
// Send error as special token
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
return len(text) / 4
}

// Close releases model resources
func (e *LlamaEngine) Close() error {
if e.model != nil {
llamaModel := e.model.(*llama.LLama)
llamaModel.Free()
e.model = nil
}
return nil
}
