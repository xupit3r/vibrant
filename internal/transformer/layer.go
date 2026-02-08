package transformer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// TransformerLayer represents a single transformer block
// Architecture: x -> norm -> attention -> residual -> norm -> ffn -> residual
type TransformerLayer struct {
	// Layer index
	layerIdx int

	// Sub-layers
	attn *Attention
	ffn  *FeedForward

	// Layer normalization
	attnNorm *RMSNorm // Pre-attention norm
	ffnNorm  *RMSNorm // Pre-FFN norm
}

// NewTransformerLayer creates a transformer layer from GGUF weights
func NewTransformerLayer(ggufFile *gguf.GGUFFile, cfg *Config, layerIdx int) (*TransformerLayer, error) {
	// Create attention layer
	attn, err := NewAttention(ggufFile, cfg, layerIdx)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention: %w", err)
	}

	// Create feed-forward layer
	ffn, err := NewFeedForward(ggufFile, cfg, layerIdx)
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN: %w", err)
	}

	// Load normalization weights (use eager dequantization for small tensors)
	prefix := fmt.Sprintf("blk.%d", layerIdx)

	attnNormWeight, err := loadTensorEager(ggufFile, prefix+".attn_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load attn norm weight: %w", err)
	}

	ffnNormWeight, err := loadTensorEager(ggufFile, prefix+".ffn_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load ffn norm weight: %w", err)
	}

	attnNorm, err := NewRMSNorm(attnNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create attn norm: %w", err)
	}

	ffnNorm, err := NewRMSNorm(ffnNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create ffn norm: %w", err)
	}

	return &TransformerLayer{
		layerIdx: layerIdx,
		attn:     attn,
		ffn:      ffn,
		attnNorm: attnNorm,
		ffnNorm:  ffnNorm,
	}, nil
}

// Forward computes one transformer layer
// Input: [batch_size, seq_len, hidden_dim]
// positions: [seq_len] - position indices
// useCache: whether to use KV-cache
// Output: [batch_size, seq_len, hidden_dim]
func (l *TransformerLayer) Forward(x *tensor.Tensor, positions []int, useCache bool) (*tensor.Tensor, error) {
	// Attention block with residual connection
	// h = x + attn(norm(x))
	normed, err := l.attnNorm.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("attn norm failed: %w", err)
	}

	attnOut, err := l.attn.Forward(normed, positions, useCache)
	if err != nil {
		return nil, fmt.Errorf("attention failed: %w", err)
	}

	// Residual connection
	h := addTensors(x, attnOut)

	// FFN block with residual connection
	// output = h + ffn(norm(h))
	normed, err = l.ffnNorm.Forward(h)
	if err != nil {
		return nil, fmt.Errorf("ffn norm failed: %w", err)
	}

	ffnOut, err := l.ffn.Forward(normed)
	if err != nil {
		return nil, fmt.Errorf("ffn failed: %w", err)
	}

	// Residual connection
	output := addTensors(h, ffnOut)

	return output, nil
}

// Helper: add two tensors element-wise (residual connection)
func addTensors(a, b *tensor.Tensor) *tensor.Tensor {
	// Element-wise addition using tensor library
	return tensor.Add(a, b)
}

// ClearCache clears the KV-cache for this layer
func (l *TransformerLayer) ClearCache() {
	l.attn.ClearCache()
}

// CacheLen returns the current KV-cache length for this layer
func (l *TransformerLayer) CacheLen() int {
	return l.attn.cacheLen
}

// MoveToDevice moves layer weights to the specified device
func (l *TransformerLayer) MoveToDevice(device tensor.Device) error {
	// Move attention weights
	if err := l.attn.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move attention to device: %w", err)
	}

	// Move feedforward weights
	if err := l.ffn.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move FFN to device: %w", err)
	}

	// Move normalization weights
	if err := l.attnNorm.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move attn norm to device: %w", err)
	}

	if err := l.ffnNorm.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move ffn norm to device: %w", err)
	}

	return nil
}
