package transformer

import (
	"fmt"
	"math"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// FeedForward implements SwiGLU feed-forward network
// Formula: output = down(swish(gate(x)) * up(x))
// where swish(x) = x * sigmoid(x)
type FeedForward struct {
	gate *tensor.Tensor // Gate projection [hidden_dim, intermediate_dim]
	up   *tensor.Tensor // Up projection [hidden_dim, intermediate_dim]
	down *tensor.Tensor // Down projection [intermediate_dim, hidden_dim]

	hiddenDim       int
	intermediateDim int
}

// NewFeedForward creates a feed-forward layer from GGUF weights
func NewFeedForward(ggufFile *gguf.GGUFFile, cfg *Config, layerIdx int) (*FeedForward, error) {
	// Load weight tensors for this layer
	// Typical naming: "blk.{layer}.ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
	prefix := fmt.Sprintf("blk.%d.ffn", layerIdx)

	gate, err := loadTensor(ggufFile, prefix+"_gate.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load gate weight: %w", err)
	}

	up, err := loadTensor(ggufFile, prefix+"_up.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load up weight: %w", err)
	}

	down, err := loadTensor(ggufFile, prefix+"_down.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load down weight: %w", err)
	}

	return &FeedForward{
		gate:            gate,
		up:              up,
		down:            down,
		hiddenDim:       cfg.HiddenDim,
		intermediateDim: cfg.IntermediateDim,
	}, nil
}

// Forward computes SwiGLU feed-forward
// Input: [batch_size, seq_len, hidden_dim]
// Output: [batch_size, seq_len, hidden_dim]
func (f *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("FFN expects 3D input, got shape %v", shape)
	}

	batchSize := shape[0]
	seqLen := shape[1]

	// Gate projection
	gateProj := matmul2D(x, f.gate) // [batch, seq, intermediate_dim]

	// Up projection
	upProj := matmul2D(x, f.up) // [batch, seq, intermediate_dim]

	// Apply SwiGLU: swish(gate) * up
	// swish(x) = x * sigmoid(x)
	swiglu := tensor.NewTensor([]int{batchSize, seqLen, f.intermediateDim}, tensor.Float32)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for i := 0; i < f.intermediateDim; i++ {
				g := float64(gateProj.At(b, s, i))
				u := float64(upProj.At(b, s, i))

				// Swish activation
				swish := g * sigmoid(g)

				// SwiGLU: swish * up
				val := swish * u

				swiglu.Set(float32(val), b, s, i)
			}
		}
	}

	// Down projection
	output := matmul2D(swiglu, f.down) // [batch, seq, hidden_dim]

	return output, nil
}

// sigmoid computes sigmoid activation
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
