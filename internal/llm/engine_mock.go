// +build !llama

package llm

import (
	"context"
	"fmt"
	"time"
)

// MockEngine is a mock LLM engine for testing without llama.cpp
type MockEngine struct {
	modelPath string
	options   LoadOptions
}

// NewLlamaEngine creates a mock engine when llama.cpp is not available
func NewLlamaEngine(modelPath string, opts LoadOptions) (*LlamaEngine, error) {
	return &LlamaEngine{
		model:   nil,
		options: opts,
	}, nil
}

// Generate generates mock text
func (e *LlamaEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
	// Simulate thinking time
	time.Sleep(500 * time.Millisecond)
	
	response := fmt.Sprintf(`[MOCK RESPONSE - llama.cpp not compiled]

This is a mock response. The actual LLM inference requires:
1. Building llama.cpp with CGO
2. Downloading a model
3. Loading the model into memory

Your question: %s

To enable real inference:
- Install build-essential (Linux) or Xcode tools (macOS)
- Run: CGO_ENABLED=1 go build -tags llama
- Use with a real downloaded model

Context size: %d tokens
Threads: %d`, prompt, e.options.ContextSize, e.options.Threads)
	
	return response, nil
}

// GenerateStream generates mock streaming text
func (e *LlamaEngine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
	ch := make(chan string, 10)
	
	go func() {
		defer close(ch)
		
		response := fmt.Sprintf(`[MOCK STREAMING RESPONSE]

This is a mock streaming response.
The actual LLM would stream tokens here.

Your question: %s

To enable real inference, llama.cpp needs to be compiled.`, prompt)
		
		// Simulate streaming
		words := []rune(response)
		for i := 0; i < len(words); i += 5 {
			select {
			case <-ctx.Done():
				return
			case ch <- string(words[i:min(i+5, len(words))]):
				time.Sleep(50 * time.Millisecond)
			}
		}
	}()
	
	return ch, nil
}

// TokenCount estimates token count
func (e *LlamaEngine) TokenCount(text string) int {
	return len(text) / 4
}

// Close does nothing for mock
func (e *LlamaEngine) Close() error {
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
