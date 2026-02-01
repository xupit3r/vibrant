package transformer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// Model represents a complete transformer language model
type Model struct {
	config *Config

	// Token embeddings
	embeddings *Embeddings

	// Transformer layers
	layers []*TransformerLayer

	// Output layer normalization
	outputNorm *RMSNorm

	// Output projection (language modeling head)
	outputWeight *tensor.Tensor // [hidden_dim, vocab_size]
}

// NewModel creates a complete model from GGUF file
func NewModel(ggufFile *gguf.GGUFFile) (*Model, error) {
	// Load configuration from GGUF metadata
	cfg, err := NewConfigFromGGUF(ggufFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	fmt.Printf("Loaded config: %s\n", cfg.String())

	// Create embeddings layer
	embeddings, err := NewEmbeddings(ggufFile, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	// Create all transformer layers
	layers := make([]*TransformerLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layer, err := NewTransformerLayer(ggufFile, cfg, i)
		if err != nil {
			return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	// Load output normalization (use eager dequantization for small tensors)
	outputNormWeight, err := loadTensorEager(ggufFile, "output_norm.weight")
	if err != nil {
		// Try alternative name
		outputNormWeight, err = loadTensorEager(ggufFile, "norm.weight")
		if err != nil {
			return nil, fmt.Errorf("failed to load output norm weight: %w", err)
		}
	}

	outputNorm, err := NewRMSNorm(outputNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create output norm: %w", err)
	}

	// Load output projection (LM head) - keep lazy for large matrix
	outputWeight, err := loadTensor(ggufFile, "output.weight")
	if err != nil {
		// Some models tie weights with embeddings
		outputWeight, err = loadTensor(ggufFile, "token_embd.weight")
		if err != nil {
			return nil, fmt.Errorf("failed to load output weight: %w", err)
		}
	}

	return &Model{
		config:       cfg,
		embeddings:   embeddings,
		layers:       layers,
		outputNorm:   outputNorm,
		outputWeight: outputWeight,
	}, nil
}

// Forward performs a forward pass through the entire model
// Input: token IDs [batch_size, seq_len]
// Output: logits [batch_size, seq_len, vocab_size]
func (m *Model) Forward(tokenIDs [][]int, useCache bool) (*tensor.Tensor, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("empty token IDs")
	}

	seqLen := len(tokenIDs[0])

	// Create position indices
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}

	// 1. Embed tokens
	hidden, err := m.embeddings.Forward(tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("embeddings failed: %w", err)
	}

	// 2. Pass through all transformer layers
	for i, layer := range m.layers {
		hidden, err = layer.Forward(hidden, positions, useCache)
		if err != nil {
			return nil, fmt.Errorf("layer %d failed: %w", i, err)
		}
	}

	// 3. Final normalization
	hidden, err = m.outputNorm.Forward(hidden)
	if err != nil {
		return nil, fmt.Errorf("output norm failed: %w", err)
	}

	// 4. Project to vocabulary (compute logits)
	// hidden: [batch, seq, hidden_dim], output_weight: [hidden_dim, vocab_size]
	// logits = hidden @ output_weight -> [batch, seq, vocab_size]
	shape := hidden.Shape()
	batchSize := shape[0]
	hiddenDim := shape[2]

	hiddenFlat := tensor.Reshape(hidden, []int{batchSize * seqLen, hiddenDim})
	logitsFlat := tensor.MatMul(hiddenFlat, m.outputWeight)
	logits := tensor.Reshape(logitsFlat, []int{batchSize, seqLen, m.config.VocabSize})

	return logits, nil
}

// Config returns the model configuration
func (m *Model) Config() *Config {
	return m.config
}

// NumLayers returns the number of transformer layers
func (m *Model) NumLayers() int {
	return len(m.layers)
}

// ClearCache clears the KV-cache for all layers
func (m *Model) ClearCache() {
	for _, layer := range m.layers {
		layer.ClearCache()
	}
}
