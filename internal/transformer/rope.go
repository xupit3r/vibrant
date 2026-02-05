package transformer

import (
	"fmt"
	"math"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// RoPE implements Rotary Positional Embeddings
type RoPE struct {
	cosTable []float32 // Precomputed cos values [maxSeqLen * halfDim]
	sinTable []float32 // Precomputed sin values [maxSeqLen * halfDim]
	dim      int       // Head dimension
	halfDim  int       // Half of head dimension
}

// NewRoPE creates a RoPE layer with precomputed cos/sin lookup tables
func NewRoPE(headDim int, freqBase float64, maxSeqLen int) *RoPE {
	halfDim := headDim / 2
	tableSize := maxSeqLen * halfDim

	cosTable := make([]float32, tableSize)
	sinTable := make([]float32, tableSize)

	// Precompute cos/sin for all positions and dimension pairs
	for pos := 0; pos < maxSeqLen; pos++ {
		base := pos * halfDim
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(freqBase, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			cosTable[base+i] = float32(math.Cos(angle))
			sinTable[base+i] = float32(math.Sin(angle))
		}
	}

	return &RoPE{
		cosTable: cosTable,
		sinTable: sinTable,
		dim:      headDim,
		halfDim:  halfDim,
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

	// Try GPU path first
	if x.IsOnGPU() {
		output := tensor.RoPEGPU(x, r.cosTable, r.sinTable, positions)
		if output != nil {
			return output, nil
		}
		// Fall through to CPU if GPU fails - need to transfer to CPU first
	}

	// CPU fallback
	// If input is on GPU, transfer to CPU first
	var xCPU *tensor.Tensor
	if x.IsOnGPU() {
		var err error
		xCPU, err = x.ToDevice(tensor.CPU)
		if err != nil {
			return nil, fmt.Errorf("failed to transfer GPU tensor to CPU for RoPE: %w", err)
		}
	} else {
		xCPU = x
	}

	// Create output tensor on CPU
	output := tensor.NewTensor(shape, tensor.Float32)

	// Direct slice access — no At()/Set() calls
	xData := xCPU.Data().([]float32)  // Use xCPU instead of x
	outData := output.Data().([]float32)

	halfDim := r.halfDim
	cosTable := r.cosTable
	sinTable := r.sinTable

	// Precompute strides
	headStride := seqLen * headDim
	batchStride := numHeads * headStride

	for b := 0; b < batchSize; b++ {
		bOff := b * batchStride
		for h := 0; h < numHeads; h++ {
			hOff := bOff + h*headStride
			for s := 0; s < seqLen; s++ {
				sOff := hOff + s*headDim
				tOff := positions[s] * halfDim

				for i := 0; i < halfDim; i++ {
					c := cosTable[tOff+i]
					sn := sinTable[tOff+i]

					x0 := xData[sOff+2*i]
					x1 := xData[sOff+2*i+1]

					outData[sOff+2*i] = x0*c - x1*sn
					outData[sOff+2*i+1] = x0*sn + x1*c
				}
			}
		}
	}

	// If input was on GPU, transfer output back to GPU
	if x.IsOnGPU() {
		outputGPU, err := output.ToDevice(tensor.GPU)
		if err != nil {
			return nil, fmt.Errorf("failed to transfer RoPE output back to GPU: %w", err)
		}
		return outputGPU, nil
	}

	return output, nil
}
