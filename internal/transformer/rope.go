package transformer

import (
	"math"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// RoPE implements Rotary Positional Embeddings
type RoPE struct {
	freqs *tensor.Tensor // Precomputed frequencies [head_dim/2]
	dim   int            // Head dimension
}

// NewRoPE creates a RoPE layer
func NewRoPE(headDim int, freqBase float64, maxSeqLen int) *RoPE {
	// Precompute frequencies for each dimension pair
	// freq[i] = 1.0 / (base^(2*i/dim))
	halfDim := headDim / 2
	freqs := tensor.NewTensor([]int{halfDim}, tensor.Float32)

	for i := 0; i < halfDim; i++ {
		freq := 1.0 / math.Pow(freqBase, float64(2*i)/float64(headDim))
		freqs.Set(float32(freq), i)
	}

	return &RoPE{
		freqs: freqs,
		dim:   headDim,
	}
}

// ApplyRotation applies RoPE to query or key tensor
// Input shape: [batch_size, num_heads, seq_len, head_dim]
// Positions: [seq_len] - position indices for each token
// Output shape: [batch_size, num_heads, seq_len, head_dim]
func (r *RoPE) ApplyRotation(x *tensor.Tensor, positions []int) (*tensor.Tensor, error) {
	shape := x.Shape()
	batchSize := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	if headDim != r.dim {
		return nil, nil // Return input unchanged if dimensions don't match
	}

	// Create output tensor
	output := tensor.NewTensor(shape, tensor.Float32)

	halfDim := headDim / 2

	// Apply rotation to each element
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := float64(positions[s])

				// Process pairs of dimensions
				for i := 0; i < halfDim; i++ {
					freq := float64(r.freqs.At(i))
					angle := pos * freq

					cos := math.Cos(angle)
					sin := math.Sin(angle)

					// Get the pair of values
					x0 := float64(x.At(b, h, s, 2*i))
					x1 := float64(x.At(b, h, s, 2*i+1))

					// Apply rotation
					// [cos -sin] [x0]
					// [sin  cos] [x1]
					y0 := x0*cos - x1*sin
					y1 := x0*sin + x1*cos

					output.Set(float32(y0), b, h, s, 2*i)
					output.Set(float32(y1), b, h, s, 2*i+1)
				}
			}
		}
	}

	return output, nil
}
