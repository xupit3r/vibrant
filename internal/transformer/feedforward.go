package transformer

import (
	"fmt"

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

	// Pre-transpose Float32 weight matrices for matmul optimization
	// This is done once at load time instead of 168-224 times per forward pass
	// For quantized weights, transpose happens during dequantization (handled by cache)
	if gate.DType() == tensor.Float32 {
		if err := gate.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose gate: %w", err)
		}
	}
	if up.DType() == tensor.Float32 {
		if err := up.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose up: %w", err)
		}
	}
	if down.DType() == tensor.Float32 {
		if err := down.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose down: %w", err)
		}
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

	// Flatten batch dimension for matmul: [batch*seq, hidden_dim]
	xFlat := tensor.Reshape(x, []int{batchSize * seqLen, f.hiddenDim})

	// Gate projection: [batch*seq, intermediate_dim]
	gateProjFlat := tensor.MatMul(xFlat, f.gate)

	// Up projection: [batch*seq, intermediate_dim]
	upProjFlat := tensor.MatMul(xFlat, f.up)

	// Apply SwiGLU: SiLU(gate) * up
	// This uses GPU-accelerated SiLU and Mul operations when tensors are on GPU
	gateActivated := tensor.SiLU(gateProjFlat)
	swiGLUResult := tensor.Mul(gateActivated, upProjFlat)

	// Down projection: [batch*seq, hidden_dim]
	outputFlat := tensor.MatMul(swiGLUResult, f.down)
	output := tensor.Reshape(outputFlat, []int{batchSize, seqLen, f.hiddenDim})

	return output, nil
}

// MoveToDevice moves feedforward weights to the specified device
func (f *FeedForward) MoveToDevice(device tensor.Device) error {
	// Move gate, up, down weights to device
	weights := []*tensor.Tensor{f.gate, f.up, f.down}
	ptrs := []*(*tensor.Tensor){&f.gate, &f.up, &f.down}
	names := []string{"gate", "up", "down"}

	for i, w := range weights {
		if w == nil {
			continue
		}
		gpuWeight, err := w.ToDevice(device)
		if err != nil {
			return fmt.Errorf("failed to move %s to device: %w", names[i], err)
		}
		w.FreeGPU() // Free old GPU memory if any
		*ptrs[i] = gpuWeight
	}

	return nil
}
