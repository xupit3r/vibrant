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

	// Pre-transpose Float32 weight matrices for matmul optimization
	// This is done once at load time instead of 168-224 times per forward pass
	// For quantized weights, transpose happens during dequantization (handled by cache)
	if wq.DType() == tensor.Float32 {
		if err := wq.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose wq: %w", err)
		}
	}
	if wk.DType() == tensor.Float32 {
		if err := wk.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose wk: %w", err)
		}
	}
	if wv.DType() == tensor.Float32 {
		if err := wv.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose wv: %w", err)
		}
	}
	if wo.DType() == tensor.Float32 {
		if err := wo.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose wo: %w", err)
		}
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

	// If using cache, concatenate with cached K/V
	if useCache {
		if a.kCache != nil && a.cacheLen > 0 {
			// Concatenate cached K/V with new K/V along sequence dimension
			// Cached: [batch, num_kv_heads, cache_len, head_dim]
			// New: [batch, num_kv_heads, seq_len, head_dim]
			// Result: [batch, num_kv_heads, cache_len + seq_len, head_dim]
			k = concatenateSeqDim(a.kCache, k, a.cacheLen)
			v = concatenateSeqDim(a.vCache, v, a.cacheLen)
		}

		// Update cache with full K/V (including new tokens)
		// Store for next iteration
		a.kCache = k
		a.vCache = v
		a.cacheLen = k.Shape()[2] // seq dimension
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

// Helper: load tensor with lazy dequantization
// This keeps large weight matrices (attention, FFN) in Q5_K format
// and will be dequantized on-demand during MatMul operations
func loadTensor(ggufFile *gguf.GGUFFile, name string) (*tensor.Tensor, error) {
	t, err := ggufFile.LoadTensor(name)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// Helper: load tensor with eager dequantization
// Use this for small, frequently accessed tensors (norms, biases)
// that need to be in Float32 format
func loadTensorEager(ggufFile *gguf.GGUFFile, name string) (*tensor.Tensor, error) {
	t, err := ggufFile.LoadTensor(name)
	if err != nil {
		return nil, err
	}

	// Eagerly dequantize Q5_K tensors to Float32
	if t.DType() == tensor.Q5_K {
		dequantized, err := tensor.DequantizeQ5_KTensor(t)
		if err != nil {
			return nil, fmt.Errorf("failed to dequantize %s: %w", name, err)
		}
		return dequantized, nil
	}

	return t, nil
}

// transposeHeads transposes from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
func transposeHeads(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shape()
	batch, seq, heads, headDim := shape[0], shape[1], shape[2], shape[3]

	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if x.IsOnGPU() {
		if _, err := x.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("transposeHeads: failed to get CPU data: %v", err))
		}
	}

	result := tensor.NewTensor([]int{batch, heads, seq, headDim}, x.DType())
	src := x.Data().([]float32)
	dst := result.Data().([]float32)

	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for h := 0; h < heads; h++ {
				srcOff := b*seq*heads*headDim + s*heads*headDim + h*headDim
				dstOff := b*heads*seq*headDim + h*seq*headDim + s*headDim
				copy(dst[dstOff:dstOff+headDim], src[srcOff:srcOff+headDim])
			}
		}
	}

	return result
}

// transposeHeadsBack transposes from [batch, heads, seq, head_dim] to [batch, seq, heads, head_dim]
func transposeHeadsBack(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shape()
	batch, heads, seq, headDim := shape[0], shape[1], shape[2], shape[3]

	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if x.IsOnGPU() {
		if _, err := x.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("transposeHeadsBack: failed to get CPU data: %v", err))
		}
	}

	result := tensor.NewTensor([]int{batch, seq, heads, headDim}, x.DType())
	src := x.Data().([]float32)
	dst := result.Data().([]float32)

	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seq; s++ {
				srcOff := b*heads*seq*headDim + h*seq*headDim + s*headDim
				dstOff := b*seq*heads*headDim + s*heads*headDim + h*headDim
				copy(dst[dstOff:dstOff+headDim], src[srcOff:srcOff+headDim])
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

	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if kv.IsOnGPU() {
		if _, err := kv.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("expandKVHeads: failed to get CPU data: %v", err))
		}
	}

	groupSize := numHeads / numKVHeads
	result := tensor.NewTensor([]int{batch, numHeads, seq, headDim}, kv.DType())
	src := kv.Data().([]float32)
	dst := result.Data().([]float32)
	seqHeadDim := seq * headDim

	for b := 0; b < batch; b++ {
		for kvh := 0; kvh < numKVHeads; kvh++ {
			srcOff := b*numKVHeads*seqHeadDim + kvh*seqHeadDim
			for g := 0; g < groupSize; g++ {
				h := kvh*groupSize + g
				dstOff := b*numHeads*seqHeadDim + h*seqHeadDim
				copy(dst[dstOff:dstOff+seqHeadDim], src[srcOff:srcOff+seqHeadDim])
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
	seq := shape[2]
	headDim := shape[3]
	heads := shape[1]

	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if t.IsOnGPU() {
		if _, err := t.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("extractSlice: failed to get CPU data: %v", err))
		}
	}

	result := tensor.NewTensor([]int{seq, headDim}, t.DType())
	src := t.Data().([]float32)
	dst := result.Data().([]float32)
	srcBase := batch*heads*seq*headDim + head*seq*headDim
	copy(dst, src[srcBase:srcBase+seq*headDim])
	return result
}

// copySlice copies a [seq, head_dim] slice back to [batch, heads, seq, head_dim]
func copySlice(src *tensor.Tensor, dst *tensor.Tensor, batch, head int) {
	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if src.IsOnGPU() {
		if _, err := src.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("copySlice: failed to get src CPU data: %v", err))
		}
	}
	if dst.IsOnGPU() {
		if _, err := dst.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("copySlice: failed to get dst CPU data: %v", err))
		}
	}

	shape := src.Shape()
	seq := shape[0]
	headDim := shape[1]
	dstShape := dst.Shape()
	heads := dstShape[1]

	srcData := src.Data().([]float32)
	dstData := dst.Data().([]float32)
	dstBase := batch*heads*seq*headDim + head*seq*headDim
	copy(dstData[dstBase:dstBase+seq*headDim], srcData)
}

// scaleScores scales all elements in the tensor by a factor
func scaleScores(scores *tensor.Tensor, scale float64) {
	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if scores.IsOnGPU() {
		if _, err := scores.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("scaleScores: failed to get CPU data: %v", err))
		}
	}

	data := scores.Data().([]float32)
	s := float32(scale)
	for i := range data {
		data[i] *= s
	}
}

// applyCausalMask applies causal masking to prevent attending to future positions
func applyCausalMask(scores *tensor.Tensor) {
	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if scores.IsOnGPU() {
		if _, err := scores.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("applyCausalMask: failed to get CPU data: %v", err))
		}
	}

	shape := scores.Shape()
	rows := shape[0]
	cols := shape[1]
	data := scores.Data().([]float32)
	negInf := float32(math.Inf(-1))

	for i := 0; i < rows; i++ {
		for j := i + 1; j < cols; j++ {
			data[i*cols+j] = negInf
		}
	}
}

// applySoftmax applies softmax to each row of the scores matrix
func applySoftmax(scores *tensor.Tensor) {
	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if scores.IsOnGPU() {
		if _, err := scores.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("applySoftmax: failed to get CPU data: %v", err))
		}
	}

	shape := scores.Shape()
	rows := shape[0]
	cols := shape[1]
	data := scores.Data().([]float32)

	for i := 0; i < rows; i++ {
		row := data[i*cols : (i+1)*cols]

		// Find max for numerical stability
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		// Compute exp and sum
		sum := float32(0)
		for j := range row {
			if row[j] == float32(math.Inf(-1)) {
				row[j] = 0
			} else {
				row[j] = float32(math.Exp(float64(row[j] - maxVal)))
				sum += row[j]
			}
		}

		// Normalize
		if sum > 0 {
			invSum := 1.0 / sum
			for j := range row {
				row[j] *= invSum
			}
		}
	}
}

// concatenateSeqDim concatenates two tensors along the sequence dimension
// cached: [batch, heads, cache_len, head_dim]
// new: [batch, heads, seq_len, head_dim]
// -> [batch, heads, cache_len + seq_len, head_dim]
func concatenateSeqDim(cached, new *tensor.Tensor, cacheLen int) *tensor.Tensor {
	cachedShape := cached.Shape()
	newShape := new.Shape()

	batch := cachedShape[0]
	heads := cachedShape[1]
	newSeqLen := newShape[2]
	headDim := cachedShape[3]

	// Ensure CPU data is available (GPU tensors may have zeroed CPU copy)
	if cached.IsOnGPU() {
		if _, err := cached.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("concatenateSeqDim: failed to get cached CPU data: %v", err))
		}
	}
	if new.IsOnGPU() {
		if _, err := new.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("concatenateSeqDim: failed to get new CPU data: %v", err))
		}
	}

	totalSeqLen := cacheLen + newSeqLen
	result := tensor.NewTensor([]int{batch, heads, totalSeqLen, headDim}, cached.DType())

	cachedData := cached.Data().([]float32)
	newData := new.Data().([]float32)
	dstData := result.Data().([]float32)

	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			// Copy cached portion
			cachedOff := b*heads*cachedShape[2]*headDim + h*cachedShape[2]*headDim
			dstOff := b*heads*totalSeqLen*headDim + h*totalSeqLen*headDim
			copy(dstData[dstOff:dstOff+cacheLen*headDim], cachedData[cachedOff:cachedOff+cacheLen*headDim])

			// Copy new portion
			newOff := b*heads*newSeqLen*headDim + h*newSeqLen*headDim
			dstNewOff := dstOff + cacheLen*headDim
			copy(dstData[dstNewOff:dstNewOff+newSeqLen*headDim], newData[newOff:newOff+newSeqLen*headDim])
		}
	}

	return result
}

// ClearCache clears the KV-cache for this attention layer
func (a *Attention) ClearCache() {
	a.kCache = nil
	a.vCache = nil
	a.cacheLen = 0
}

// MoveToDevice moves attention weights to the specified device
func (a *Attention) MoveToDevice(device tensor.Device) error {
	// Move Q, K, V, O weights to device
	weights := []*tensor.Tensor{a.wq, a.wk, a.wv, a.wo}
	ptrs := []*(*tensor.Tensor){&a.wq, &a.wk, &a.wv, &a.wo}
	names := []string{"wq", "wk", "wv", "wo"}

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
