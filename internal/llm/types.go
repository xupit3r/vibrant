package llm

import (
	"context"
	"runtime"
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
	MaxTokens   int      // Maximum tokens to generate
	Temperature float32  // Randomness (0.0-1.0)
	TopP        float32  // Nucleus sampling
	TopK        int      // Top-k sampling
	StopTokens  []string // Stop sequences
}

// LoadOptions configures model loading
type LoadOptions struct {
	ContextSize int    // Max context tokens
	Threads     int    // CPU threads for inference
	BatchSize   int    // Batch size for prompt processing
	UseMMap     bool   // Use memory-mapped files
	UseMlock    bool   // Lock pages in RAM
	Verbose     bool   // Enable verbose logging
	Device      string // Device to use: "cpu", "gpu", or "auto"
}

// LlamaEngine wraps the underlying LLM engine
type LlamaEngine struct {
	model   interface{} // Actual type depends on build tags
	options LoadOptions
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
