package transformer

import (
	"fmt"
	"math"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// Attention implements multi-head self-attention with optional KV-cache
type Attention struct {
	// Projection weights
	wq *tensor.Tensor // Query projection [hidden_dim, num_heads * head_dim]
	wk *tensor.Tensor // Key projection [hidden_dim, num_kv_heads * head_dim]
	wv *tensor.Tensor // Value projection [hidden_dim, num_kv_heads * head_dim]
	wo *tensor.Tensor // Output projection [num_heads * head_dim, hidden_dim]

	// Configuration
	numHeads   int
	numKVHeads int
	headDim    int
	hiddenDim  int

	// RoPE for positional encoding
	rope *RoPE

	// KV-cache for auto-regressive generation
	// Shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
	kCache *tensor.Tensor
	vCache *tensor.Tensor
	cacheLen int // Current cache length
}

// NewAttention creates an attention layer from GGUF weights
func NewAttention(ggufFile *gguf.GGUFFile, cfg *Config, layerIdx int) (*Attention, error) {
	// Load weight tensors for this layer
	// Typical naming: "blk.{layer}.attn_q.weight"
	prefix := fmt.Sprintf("blk.%d.attn", layerIdx)

	wq, err := loadTensor(ggufFile, prefix+"_q.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load Q weight: %w", err)
	}

	wk, err := loadTensor(ggufFile, prefix+"_k.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load K weight: %w", err)
	}

	wv, err := loadTensor(ggufFile, prefix+"_v.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load V weight: %w", err)
	}

	wo, err := loadTensor(ggufFile, prefix+"_output.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load output weight: %w", err)
	}

	// Create RoPE layer
	rope := NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength)

	return &Attention{
		wq:         wq,
		wk:         wk,
		wv:         wv,
		wo:         wo,
		numHeads:   cfg.NumHeads,
		numKVHeads: cfg.NumKVHeads,
		headDim:    cfg.HeadDim,
		hiddenDim:  cfg.HiddenDim,
		rope:       rope,
		cacheLen:   0,
	}, nil
}

// Forward computes multi-head attention
// Input: [batch_size, seq_len, hidden_dim]
// positions: [seq_len] - position indices
// useCache: whether to use/update KV-cache
// Output: [batch_size, seq_len, hidden_dim]
func (a *Attention) Forward(x *tensor.Tensor, positions []int, useCache bool) (*tensor.Tensor, error) {
	shape := x.Shape()
	batchSize := shape[0]
	seqLen := shape[1]

	// Project to Q, K, V using matrix multiplication
	// Input: [batch, seq, hidden_dim], Weights: [hidden_dim, proj_dim]
	// We need to reshape input to 2D, multiply, then reshape back to 3D

	// Flatten batch dimension for matmul: [batch*seq, hidden_dim]
	xFlat := tensor.Reshape(x, []int{batchSize * seqLen, a.hiddenDim})

	// Q projection: [batch*seq, num_heads * head_dim]
	qFlat := tensor.MatMul(xFlat, a.wq)
	q := tensor.Reshape(qFlat, []int{batchSize, seqLen, a.numHeads * a.headDim})

	// K projection: [batch*seq, num_kv_heads * head_dim]
	kFlat := tensor.MatMul(xFlat, a.wk)
	k := tensor.Reshape(kFlat, []int{batchSize, seqLen, a.numKVHeads * a.headDim})

	// V projection: [batch*seq, num_kv_heads * head_dim]
	vFlat := tensor.MatMul(xFlat, a.wv)
	v := tensor.Reshape(vFlat, []int{batchSize, seqLen, a.numKVHeads * a.headDim})

	// Reshape to separate heads
	// Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
	q = tensor.Reshape(q, []int{batchSize, seqLen, a.numHeads, a.headDim})
	q = transposeHeads(q) // [batch, num_heads, seq, head_dim]

	// K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
	k = tensor.Reshape(k, []int{batchSize, seqLen, a.numKVHeads, a.headDim})
	k = transposeHeads(k)

	v = tensor.Reshape(v, []int{batchSize, seqLen, a.numKVHeads, a.headDim})
	v = transposeHeads(v)

	// Apply RoPE to Q and K
	q, _ = a.rope.ApplyRotation(q, positions)
	k, _ = a.rope.ApplyRotation(k, positions)

	// If using cache, update and concatenate
	if useCache {
		// For now, simple implementation without cache
		// TODO: Add KV-cache support in future iteration
	}

	// Expand K and V for Grouped-Query Attention
	if a.numKVHeads < a.numHeads {
		k = expandKVHeads(k, a.numHeads, a.numKVHeads)
		v = expandKVHeads(v, a.numHeads, a.numKVHeads)
	}

	// Compute attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
	attn, err := computeAttention(q, k, v, a.headDim)
	if err != nil {
		return nil, err
	}

	// Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
	attn = transposeHeadsBack(attn) // [batch, seq, num_heads, head_dim]
	attn = tensor.Reshape(attn, []int{batchSize, seqLen, a.numHeads * a.headDim})

	// Output projection
	attnFlat := tensor.Reshape(attn, []int{batchSize * seqLen, a.numHeads * a.headDim})
	outputFlat := tensor.MatMul(attnFlat, a.wo)
	output := tensor.Reshape(outputFlat, []int{batchSize, seqLen, a.hiddenDim})

	return output, nil
}

// Helper: load tensor with error handling
func loadTensor(ggufFile *gguf.GGUFFile, name string) (*tensor.Tensor, error) {
	t, err := ggufFile.LoadTensor(name)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// transposeHeads transposes from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
func transposeHeads(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shape()
	batch, seq, heads, headDim := shape[0], shape[1], shape[2], shape[3]

	result := tensor.NewTensor([]int{batch, heads, seq, headDim}, x.DType())

	// Transpose the data
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for h := 0; h < heads; h++ {
				for d := 0; d < headDim; d++ {
					val := x.At(b, s, h, d)
					result.Set(val, b, h, s, d)
				}
			}
		}
	}

	return result
}

// transposeHeadsBack transposes from [batch, heads, seq, head_dim] to [batch, seq, heads, head_dim]
func transposeHeadsBack(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shape()
	batch, heads, seq, headDim := shape[0], shape[1], shape[2], shape[3]

	result := tensor.NewTensor([]int{batch, seq, heads, headDim}, x.DType())

	// Transpose the data
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seq; s++ {
				for d := 0; d < headDim; d++ {
					val := x.At(b, h, s, d)
					result.Set(val, b, s, h, d)
				}
			}
		}
	}

	return result
}

// Helper: expand KV heads for GQA
func expandKVHeads(kv *tensor.Tensor, numHeads, numKVHeads int) *tensor.Tensor {
	// Replicate each KV head for multiple Q heads
	// [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
	shape := kv.Shape()
	batch := shape[0]
	seq := shape[2]
	headDim := shape[3]

	groupSize := numHeads / numKVHeads
	result := tensor.NewTensor([]int{batch, numHeads, seq, headDim}, kv.DType())

	for b := 0; b < batch; b++ {
		for kvh := 0; kvh < numKVHeads; kvh++ {
			for s := 0; s < seq; s++ {
				for d := 0; d < headDim; d++ {
					val := kv.At(b, kvh, s, d)
					// Replicate this KV head to multiple query heads
					for g := 0; g < groupSize; g++ {
						h := kvh*groupSize + g
						result.Set(val, b, h, s, d)
					}
				}
			}
		}
	}

	return result
}

// Helper: compute attention scores and apply to values
func computeAttention(q, k, v *tensor.Tensor, headDim int) (*tensor.Tensor, error) {
	// Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
	// Input shapes: [batch, heads, seq, head_dim]

	shape := q.Shape()
	batch := shape[0]
	heads := shape[1]
	seqLen := shape[2]

	scale := 1.0 / math.Sqrt(float64(headDim))

	// Result: [batch, heads, seq, head_dim]
	result := tensor.NewTensor([]int{batch, heads, seqLen, headDim}, tensor.Float32)

	// Process each batch and head independently
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			// Extract Q, K, V for this batch and head: [seq, head_dim]
			qSlice := extractSlice(q, b, h) // [seq, head_dim]
			kSlice := extractSlice(k, b, h) // [seq, head_dim]
			vSlice := extractSlice(v, b, h) // [seq, head_dim]

			// Compute scores: Q @ K^T / sqrt(head_dim)
			// Q: [seq, head_dim], K^T: [head_dim, seq] -> scores: [seq, seq]
			kT := tensor.Transpose(kSlice)
			scores := tensor.MatMul(qSlice, kT)

			// Scale scores
			scaleScores(scores, scale)

			// Apply causal mask (prevent attending to future positions)
			applyCausalMask(scores)

			// Apply softmax row-wise
			applySoftmax(scores)

			// Multiply by V: scores @ V
			// scores: [seq, seq], V: [seq, head_dim] -> output: [seq, head_dim]
			output := tensor.MatMul(scores, vSlice)

			// Copy result back
			copySlice(output, result, b, h)
		}
	}

	return result, nil
}

// extractSlice extracts a [seq, head_dim] slice from [batch, heads, seq, head_dim]
func extractSlice(t *tensor.Tensor, batch, head int) *tensor.Tensor {
	shape := t.Shape()
	seqLen := shape[2]
	headDim := shape[3]

	result := tensor.NewTensor([]int{seqLen, headDim}, t.DType())
	for s := 0; s < seqLen; s++ {
		for d := 0; d < headDim; d++ {
			val := t.At(batch, head, s, d)
			result.Set(val, s, d)
		}
	}
	return result
}

// copySlice copies a [seq, head_dim] slice back to [batch, heads, seq, head_dim]
func copySlice(src *tensor.Tensor, dst *tensor.Tensor, batch, head int) {
	shape := src.Shape()
	seqLen := shape[0]
	headDim := shape[1]

	for s := 0; s < seqLen; s++ {
		for d := 0; d < headDim; d++ {
			val := src.At(s, d)
			dst.Set(val, batch, head, s, d)
		}
	}
}

// scaleScores scales all elements in the tensor by a factor
func scaleScores(scores *tensor.Tensor, scale float64) {
	shape := scores.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			val := float64(scores.At(i, j))
			scores.Set(float32(val*scale), i, j)
		}
	}
}

// applyCausalMask applies causal masking to prevent attending to future positions
func applyCausalMask(scores *tensor.Tensor) {
	shape := scores.Shape()
	seqLen := shape[0]

	// Set future positions to -inf
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			scores.Set(float32(math.Inf(-1)), i, j)
		}
	}
}

// applySoftmax applies softmax to each row of the scores matrix
func applySoftmax(scores *tensor.Tensor) {
	shape := scores.Shape()
	seqLen := shape[0]

	for i := 0; i < seqLen; i++ {
		// Find max for numerical stability
		maxVal := float32(math.Inf(-1))
		for j := 0; j < seqLen; j++ {
			val := scores.At(i, j)
			if val > maxVal && !math.IsInf(float64(val), -1) {
				maxVal = val
			}
		}

		// Compute exp(x - max) and sum
		sum := float32(0)
		expVals := make([]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			val := scores.At(i, j)
			if math.IsInf(float64(val), -1) {
				expVals[j] = 0
			} else {
				expVals[j] = float32(math.Exp(float64(val - maxVal)))
			}
			sum += expVals[j]
		}

		// Normalize
		for j := 0; j < seqLen; j++ {
			if sum > 0 {
				scores.Set(expVals[j]/sum, i, j)
			} else {
				scores.Set(0, i, j)
			}
		}
	}
}
