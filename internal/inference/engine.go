package inference

import (
	"context"
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
	"github.com/xupit3r/vibrant/internal/tokenizer"
	"github.com/xupit3r/vibrant/internal/transformer"
)

// Engine is the main inference engine for LLM generation
type Engine struct {
	model     *transformer.Model
	tokenizer *tokenizer.Tokenizer
	sampler   *Sampler
	config    *Config
}

// Config holds inference configuration parameters
type Config struct {
	MaxTokens   int      // Maximum number of tokens to generate
	Temperature float32  // Sampling temperature (0 = greedy, >1 = more random)
	TopP        float32  // Top-P (nucleus) sampling threshold
	TopK        int      // Top-K sampling (0 = disabled)
	StopTokens  []int    // Token IDs that stop generation
	Seed        int64    // Random seed for sampling
}

// GenerateOptions specifies per-generation configuration
type GenerateOptions struct {
	MaxTokens  int   // Override config.MaxTokens
	StopTokens []int // Override config.StopTokens
}

// NewEngine creates a new inference engine from a GGUF model file
func NewEngine(ggufPath string, config *Config) (*Engine, error) {
	// Load GGUF file
	ggufFile, err := gguf.ParseGGUF(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open GGUF file: %w", err)
	}

	// Load model
	model, err := transformer.NewModel(ggufFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Load tokenizer
	tok, err := tokenizer.NewTokenizerFromGGUF(ggufFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Create sampler
	sampler := NewSampler(config.Temperature, config.TopP, config.TopK, config.Seed)

	return &Engine{
		model:     model,
		tokenizer: tok,
		sampler:   sampler,
		config:    config,
	}, nil
}

// Generate produces text completion for the given prompt (blocking)
func (e *Engine) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
	// Clear cache for new generation
	e.model.ClearCache()

	// Tokenize prompt
	tokens := e.tokenizer.Encode(prompt, true, false) // addBOS=true, addEOS=false

	// Determine max tokens
	maxTokens := e.config.MaxTokens
	if opts.MaxTokens > 0 {
		maxTokens = opts.MaxTokens
	}

	// Determine stop tokens
	stopTokens := e.config.StopTokens
	if len(opts.StopTokens) > 0 {
		stopTokens = opts.StopTokens
	}

	// Prefill: process all prompt tokens
	// Convert tokens to 2D array: [batch=1, seq_len]
	tokenIDs := [][]int{tokens}
	logits, err := e.model.Forward(tokenIDs, true) // useCache = true
	if err != nil {
		return "", fmt.Errorf("prefill failed: %w", err)
	}

	// Extract logits for last token: [batch=1, seq=len(tokens), vocab] -> [vocab]
	logitsShape := logits.Shape()
	lastTokenLogits := extractLastTokenLogits(logits, logitsShape)

	// Decode: generate tokens one by one
	generatedTokens := []int{}
	for i := 0; i < maxTokens; i++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		// Sample next token
		nextToken := e.sampler.Sample(lastTokenLogits)
		generatedTokens = append(generatedTokens, nextToken)

		// Check for stop tokens
		if e.isStopToken(nextToken, stopTokens) {
			break
		}

		// Decode: forward pass with single token
		tokenIDs = [][]int{{nextToken}}
		logits, err = e.model.Forward(tokenIDs, true) // useCache = true
		if err != nil {
			return "", fmt.Errorf("decode failed at token %d: %w", i, err)
		}

		// Extract logits for the generated token
		logitsShape = logits.Shape()
		lastTokenLogits = extractLastTokenLogits(logits, logitsShape)
	}

	// Decode tokens to text
	text := e.tokenizer.Decode(generatedTokens, true) // skipSpecial=true

	return text, nil
}

// GenerateStream produces text completion with streaming output (non-blocking)
func (e *Engine) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
	ch := make(chan string, 16) // Buffered for throughput

	go func() {
		defer close(ch)

		// Clear cache for new generation
		e.model.ClearCache()

		// Tokenize prompt
		tokens := e.tokenizer.Encode(prompt, true, false) // addBOS=true, addEOS=false

		// Determine max tokens
		maxTokens := e.config.MaxTokens
		if opts.MaxTokens > 0 {
			maxTokens = opts.MaxTokens
		}

		// Determine stop tokens
		stopTokens := e.config.StopTokens
		if len(opts.StopTokens) > 0 {
			stopTokens = opts.StopTokens
		}

		// Prefill stage
		tokenIDs := [][]int{tokens}
		logits, err := e.model.Forward(tokenIDs, true)
		if err != nil {
			ch <- fmt.Sprintf("ERROR: prefill failed: %v", err)
			return
		}

		logitsShape := logits.Shape()
		lastTokenLogits := extractLastTokenLogits(logits, logitsShape)

		// Decode stage: stream tokens as generated
		for i := 0; i < maxTokens; i++ {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			// Sample next token
			nextToken := e.sampler.Sample(lastTokenLogits)

			// Decode token to text and send immediately
			tokenText := e.tokenizer.Decode([]int{nextToken}, true) // skipSpecial=true
			ch <- tokenText

			// Check for stop tokens
			if e.isStopToken(nextToken, stopTokens) {
				break
			}

			// Continue decoding
			tokenIDs = [][]int{{nextToken}}
			logits, err = e.model.Forward(tokenIDs, true)
			if err != nil {
				ch <- fmt.Sprintf("ERROR: decode failed: %v", err)
				return
			}

			logitsShape = logits.Shape()
			lastTokenLogits = extractLastTokenLogits(logits, logitsShape)
		}
	}()

	return ch, nil
}

// TokenCount returns the number of tokens in the given text
func (e *Engine) TokenCount(text string) int {
	tokens := e.tokenizer.Encode(text, false, false) // no special tokens for counting
	return len(tokens)
}

// Close releases resources held by the engine
func (e *Engine) Close() error {
	// Clear cache
	e.model.ClearCache()
	return nil
}

// isStopToken checks if the given token is a stop token
func (e *Engine) isStopToken(token int, stopTokens []int) bool {
	for _, stopToken := range stopTokens {
		if token == stopToken {
			return true
		}
	}
	return false
}

// extractLastTokenLogits extracts logits for the last token in the sequence
// logits: [batch, seq, vocab] -> [vocab]
func extractLastTokenLogits(logits *tensor.Tensor, shape []int) *tensor.Tensor {
	seqLen := shape[1]
	vocabSize := shape[2]

	// Extract the last token's logits from batch 0
	// [batch=1, seq, vocab] -> [vocab]
	result := tensor.NewTensor([]int{vocabSize}, tensor.Float32)

	for v := 0; v < vocabSize; v++ {
		val := logits.At(0, seqLen-1, v)
		result.Set(val, v)
	}

	return result
}
