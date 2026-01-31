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

	// Project to Q, K, V
	q := matmul2D(x, a.wq) // [batch, seq, num_heads * head_dim]
	k := matmul2D(x, a.wk) // [batch, seq, num_kv_heads * head_dim]
	v := matmul2D(x, a.wv) // [batch, seq, num_kv_heads * head_dim]

	// Reshape to separate heads
	// Q: [batch, num_heads, seq, head_dim]
	// K, V: [batch, num_kv_heads, seq, head_dim]
	q = reshapeHeads(q, batchSize, seqLen, a.numHeads, a.headDim)
	k = reshapeHeads(k, batchSize, seqLen, a.numKVHeads, a.headDim)
	v = reshapeHeads(v, batchSize, seqLen, a.numKVHeads, a.headDim)

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
	attn := computeAttention(q, k, v, a.headDim)

	// Reshape back: [batch, seq, num_heads * head_dim]
	attn = reshapeFromHeads(attn, batchSize, seqLen, a.numHeads, a.headDim)

	// Output projection
	output := matmul2D(attn, a.wo)

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

// Helper: 2D matrix multiplication (simplified for now)
func matmul2D(a, b *tensor.Tensor) *tensor.Tensor {
	// Simple implementation - will be replaced with tensor.MatMul
	shapeA := a.Shape()
	shapeB := b.Shape()

	// For now, create output with expected shape
	// Proper implementation would use tensor library's matmul
	batch := shapeA[0]
	seq := shapeA[1]
	outDim := shapeB[1]

	return tensor.NewTensor([]int{batch, seq, outDim}, tensor.Float32)
}

// Helper: reshape to separate attention heads
func reshapeHeads(x *tensor.Tensor, batch, seq, heads, headDim int) *tensor.Tensor {
	// Reshape [batch, seq, heads*headDim] -> [batch, heads, seq, headDim]
	// Simplified - actual implementation would properly reshape data
	return tensor.NewTensor([]int{batch, heads, seq, headDim}, tensor.Float32)
}

// Helper: reshape from attention heads back to flat
func reshapeFromHeads(x *tensor.Tensor, batch, seq, heads, headDim int) *tensor.Tensor {
	// Reshape [batch, heads, seq, headDim] -> [batch, seq, heads*headDim]
	return tensor.NewTensor([]int{batch, seq, heads * headDim}, tensor.Float32)
}

// Helper: expand KV heads for GQA
func expandKVHeads(kv *tensor.Tensor, numHeads, numKVHeads int) *tensor.Tensor {
	// Replicate each KV head for multiple Q heads
	// [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, headDim]
	shape := kv.Shape()
	return tensor.NewTensor([]int{shape[0], numHeads, shape[2], shape[3]}, tensor.Float32)
}

// Helper: compute attention scores and apply to values
func computeAttention(q, k, v *tensor.Tensor, headDim int) *tensor.Tensor {
	// scores = Q @ K^T / sqrt(head_dim)
	// attn = softmax(scores) @ V

	shape := q.Shape()
	batch := shape[0]
	heads := shape[1]
	seq := shape[2]
	dim := shape[3]

	// Compute scores: Q @ K^T
	// For simplicity, create output tensor
	// Actual implementation would:
	// 1. Transpose K
	// 2. Batch matrix multiply Q @ K^T
	// 3. Scale by 1/sqrt(head_dim)
	// 4. Apply causal mask
	// 5. Softmax
	// 6. Multiply by V

	scale := 1.0 / math.Sqrt(float64(headDim))
	_ = scale // Will be used in actual implementation

	// Return shape: [batch, heads, seq, head_dim]
	return tensor.NewTensor([]int{batch, heads, seq, dim}, tensor.Float32)
}
