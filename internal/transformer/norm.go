package transformer

import (
	"fmt"
	"math"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// RMSNorm implements Root Mean Square Layer Normalization
// Formula: y = x * rsqrt(mean(x²) + eps) * weight
type RMSNorm struct {
	weight *tensor.Tensor // [hidden_dim]
	eps    float64
}

// NewRMSNorm creates an RMSNorm layer with given weight and epsilon
func NewRMSNorm(weight *tensor.Tensor, eps float64) (*RMSNorm, error) {
	// Validate weight shape: should be 1D
	shape := weight.Shape()
	if len(shape) != 1 {
		return nil, fmt.Errorf("RMSNorm weight must be 1D, got shape %v", shape)
	}

	return &RMSNorm{
		weight: weight,
		eps:    eps,
	}, nil
}

// Forward applies RMSNorm to input
// Input shape: [batch_size, seq_len, hidden_dim]
// Output shape: [batch_size, seq_len, hidden_dim]
func (r *RMSNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("RMSNorm expects 3D input [batch, seq, hidden], got shape %v", shape)
	}

	batchSize := shape[0]
	seqLen := shape[1]
	hiddenDim := shape[2]

	// Verify weight dimension matches input
	if r.weight.Shape()[0] != hiddenDim {
		return nil, fmt.Errorf("weight dim %d != hidden dim %d", r.weight.Shape()[0], hiddenDim)
	}

	// Create output tensor
	output := tensor.NewTensor(shape, tensor.Float32)

	// Apply RMSNorm to each position independently
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// Compute RMS (root mean square)
			sumSquares := 0.0
			for d := 0; d < hiddenDim; d++ {
				val := float64(x.At(b, s, d))
				sumSquares += val * val
			}
			rms := math.Sqrt(sumSquares/float64(hiddenDim) + r.eps)

			// Normalize and scale
			for d := 0; d < hiddenDim; d++ {
				val := float64(x.At(b, s, d))
				weight := float64(r.weight.At(d))
				normalized := (val / rms) * weight
				output.Set(float32(normalized), b, s, d)
			}
		}
	}

	return output, nil
}
