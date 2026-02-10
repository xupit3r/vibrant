package llm

import (
	"context"
	"fmt"

	"github.com/xupit3r/vibrant/internal/inference"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// CustomEngine implements the Engine interface using our custom pure Go inference engine
type CustomEngine struct {
	engine *inference.Engine
	path   string
}

// NewCustomEngine creates a new custom inference engine from a GGUF model file
func NewCustomEngine(modelPath string, opts LoadOptions) (*CustomEngine, error) {
	// Parse device option
	var device tensor.Device
	switch opts.Device {
	case "gpu", "cuda", "metal":
		// GPU, CUDA, and Metal all map to tensor.GPU
		device = tensor.GPU
	case "cpu":
		device = tensor.CPU
	case "auto":
		// Auto: try GPU first, fall back to CPU
		device = tensor.GPU
	default:
		device = tensor.CPU
	}

	// Convert LoadOptions to inference.Config
	config := &inference.Config{
		MaxTokens:   opts.ContextSize / 2, // Reserve half for prompt
		Temperature: 0.2,                   // Default for code generation
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{}, // Will be populated from tokenizer's EOS
		Seed:        42,      // Deterministic by default
		Device:      device,
	}

	// Create inference engine
	engine, err := inference.NewEngine(modelPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create custom engine: %w", err)
	}

	return &CustomEngine{
		engine: engine,
		path:   modelPath,
	}, nil
}

// Generate produces a text completion for the given prompt (blocking)
func (e *CustomEngine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
	// Convert GenerateOptions to inference.GenerateOptions
	inferOpts := inference.GenerateOptions{
		MaxTokens: opts.MaxTokens,
		// StopTokens conversion: string -> token IDs would require tokenizer access
		// For now, we'll rely on the model's EOS token
		StopTokens: []int{},
	}

	// Update sampler settings if different from engine config
	// Note: This is a limitation - we can't change temperature per request easily
	// Future improvement: Make sampler settings per-request

	return e.engine.Generate(ctx, prompt, inferOpts)
}

// GenerateStream produces a text completion with streaming output (non-blocking)
func (e *CustomEngine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
	// Convert options
	inferOpts := inference.GenerateOptions{
		MaxTokens:  opts.MaxTokens,
		StopTokens: []int{},
	}

	return e.engine.GenerateStream(ctx, prompt, inferOpts)
}

// TokenCount returns the number of tokens in the given text
func (e *CustomEngine) TokenCount(text string) int {
	return e.engine.TokenCount(text)
}

// Close releases resources held by the engine
func (e *CustomEngine) Close() error {
	return e.engine.Close()
}

// FormatPrompt applies the model's chat template to format a system+user prompt.
// Falls back to plain text if no chat template was detected.
func (e *CustomEngine) FormatPrompt(system, user string) string {
	if tmpl := e.engine.ChatTemplate(); tmpl != nil {
		return tmpl.FormatSimple(system, user)
	}
	// Fallback: plain concatenation
	if system != "" {
		return system + "\n\n" + user
	}
	return user
}

// String returns a string representation of the engine
func (e *CustomEngine) String() string {
	return fmt.Sprintf("CustomEngine{path: %s, type: pure-go}", e.path)
}

// convertStopTokens converts string stop sequences to token IDs
// This is a placeholder - actual implementation would need tokenizer access
func (e *CustomEngine) convertStopTokens(stopStrings []string) []int {
	// TODO: Implement proper string -> token ID conversion
	// For now, return empty and rely on EOS token
	return []int{}
}

// NewCustomEngineWithConfig creates a custom engine with more control over inference settings
func NewCustomEngineWithConfig(modelPath string, opts LoadOptions, genOpts GenerateOptions) (*CustomEngine, error) {
	// Convert GenerateOptions to inference.Config for default settings
	config := &inference.Config{
		MaxTokens:   genOpts.MaxTokens,
		Temperature: genOpts.Temperature,
		TopP:        genOpts.TopP,
		TopK:        genOpts.TopK,
		StopTokens:  []int{},
		Seed:        42,
	}

	// Create inference engine
	engine, err := inference.NewEngine(modelPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create custom engine: %w", err)
	}

	return &CustomEngine{
		engine: engine,
		path:   modelPath,
	}, nil
}
