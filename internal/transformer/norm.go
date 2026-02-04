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

	xData := x.Data().([]float32)
	oData := output.Data().([]float32)
	wData := r.weight.Data().([]float32)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * hiddenDim
			row := xData[off : off+hiddenDim]
			out := oData[off : off+hiddenDim]

			// Compute sum of squares
			sumSq := float32(0)
			for _, v := range row {
				sumSq += v * v
			}

			// RMS = sqrt(mean(x^2) + eps), then compute 1/rms
			rmsInv := float32(1.0 / math.Sqrt(float64(sumSq/float32(hiddenDim))+r.eps))

			// Normalize and scale by weight
			for d := 0; d < hiddenDim; d++ {
				out[d] = row[d] * rmsInv * wData[d]
			}
		}
	}

	return output, nil
}

// MoveToDevice moves RMSNorm weight to the specified device
func (r *RMSNorm) MoveToDevice(device tensor.Device) error {
	// RMSNorm uses element-wise operations, keep weight on CPU for now
	// GPU RMSNorm can be added later for full GPU acceleration
	return nil
}
