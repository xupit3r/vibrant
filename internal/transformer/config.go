package transformer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
)

// Config holds transformer model hyperparameters
type Config struct {
	// Model architecture
	Architecture string // "qwen", "llama", etc.

	// Sequence and dimensions
	ContextLength   int // Maximum sequence length
	VocabSize       int // Vocabulary size
	HiddenDim       int // Model dimension (embedding_length)
	NumLayers       int // Number of transformer blocks
	IntermediateDim int // FFN intermediate dimension

	// Attention configuration
	NumHeads      int // Number of attention heads
	NumKVHeads    int // Number of KV heads (for GQA)
	HeadDim       int // Dimension per head (HiddenDim / NumHeads)

	// RoPE configuration
	RopeFreqBase float64 // RoPE frequency base
	RopeScaling  float64 // RoPE scaling factor (default 1.0)

	// Normalization
	RMSNormEps float64 // RMSNorm epsilon
}

// NewConfigFromGGUF creates a Config from GGUF metadata
func NewConfigFromGGUF(ggufFile *gguf.GGUFFile) (*Config, error) {
	cfg := &Config{}

	// Get architecture
	cfg.Architecture = ggufFile.GetArchitecture()
	if cfg.Architecture == "unknown" {
		return nil, fmt.Errorf("architecture not found in GGUF metadata")
	}

	// Get context length
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyContextLength); ok {
		cfg.ContextLength = val
	} else {
		return nil, fmt.Errorf("context_length not found in metadata")
	}

	// Get hidden dimension (embedding_length)
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyEmbeddingLength); ok {
		cfg.HiddenDim = val
	} else {
		return nil, fmt.Errorf("embedding_length not found in metadata")
	}

	// Get number of layers (block_count)
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyBlockCount); ok {
		cfg.NumLayers = val
	} else {
		return nil, fmt.Errorf("block_count not found in metadata")
	}

	// Get number of attention heads
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyAttentionHeadCount); ok {
		cfg.NumHeads = val
	} else {
		return nil, fmt.Errorf("attention.head_count not found in metadata")
	}

	// Get number of KV heads (for Grouped-Query Attention)
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyAttentionHeadCountKV); ok {
		cfg.NumKVHeads = val
	} else {
		// Default to same as NumHeads (standard MHA)
		cfg.NumKVHeads = cfg.NumHeads
	}

	// Calculate head dimension
	cfg.HeadDim = cfg.HiddenDim / cfg.NumHeads

	// Get FFN intermediate dimension (feed_forward_length)
	if val, ok := ggufFile.GetMetadataInt(gguf.KeyFFNLength); ok {
		cfg.IntermediateDim = val
	} else {
		// Default to 4 * HiddenDim (common for transformers)
		cfg.IntermediateDim = 4 * cfg.HiddenDim
	}

	// Get RoPE frequency base
	if val, ok := ggufFile.GetMetadataFloat(gguf.KeyRopeFreqBase); ok {
		cfg.RopeFreqBase = val
	} else {
		// Default to 10000.0 (common for many models)
		cfg.RopeFreqBase = 10000.0
	}

	// RoPE scaling (default to 1.0)
	cfg.RopeScaling = 1.0

	// Get RMSNorm epsilon
	if val, ok := ggufFile.GetMetadataFloat(gguf.KeyNormRMSEps); ok {
		cfg.RMSNormEps = val
	} else {
		// Default to 1e-6
		cfg.RMSNormEps = 1e-6
	}

	// Get vocab size from GGUF
	// This is the number of tokens in the tokenizer
	tokens := ggufFile.GetTokens()
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens found in GGUF metadata")
	}
	cfg.VocabSize = len(tokens)

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return cfg, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.ContextLength <= 0 {
		return fmt.Errorf("context_length must be positive, got %d", c.ContextLength)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab_size must be positive, got %d", c.VocabSize)
	}
	if c.HiddenDim <= 0 {
		return fmt.Errorf("hidden_dim must be positive, got %d", c.HiddenDim)
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("num_layers must be positive, got %d", c.NumLayers)
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("num_heads must be positive, got %d", c.NumHeads)
	}
	if c.NumKVHeads <= 0 {
		return fmt.Errorf("num_kv_heads must be positive, got %d", c.NumKVHeads)
	}
	if c.NumKVHeads > c.NumHeads {
		return fmt.Errorf("num_kv_heads (%d) cannot exceed num_heads (%d)", c.NumKVHeads, c.NumHeads)
	}
	if c.HiddenDim%c.NumHeads != 0 {
		return fmt.Errorf("hidden_dim (%d) must be divisible by num_heads (%d)", c.HiddenDim, c.NumHeads)
	}
	if c.IntermediateDim <= 0 {
		return fmt.Errorf("intermediate_dim must be positive, got %d", c.IntermediateDim)
	}
	if c.RopeFreqBase <= 0 {
		return fmt.Errorf("rope_freq_base must be positive, got %f", c.RopeFreqBase)
	}
	if c.RMSNormEps <= 0 {
		return fmt.Errorf("rms_norm_eps must be positive, got %e", c.RMSNormEps)
	}

	return nil
}

// IsGQA returns true if the model uses Grouped-Query Attention
func (c *Config) IsGQA() bool {
	return c.NumKVHeads < c.NumHeads
}

// KVGroupSize returns the number of Q heads per KV head
func (c *Config) KVGroupSize() int {
	return c.NumHeads / c.NumKVHeads
}

// String returns a human-readable representation of the config
func (c *Config) String() string {
	return fmt.Sprintf(
		"Config{arch=%s, ctx=%d, vocab=%d, dim=%d, layers=%d, heads=%d, kv_heads=%d, "+
			"head_dim=%d, ffn=%d, rope_base=%.1f, eps=%e}",
		c.Architecture, c.ContextLength, c.VocabSize, c.HiddenDim, c.NumLayers,
		c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateDim, c.RopeFreqBase, c.RMSNormEps,
	)
}
