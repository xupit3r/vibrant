package inference

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// KVCache manages key-value cache for transformer layers during autoregressive generation
// This enables efficient decoding by caching attention keys and values from previous tokens
type KVCache struct {
	numLayers int                  // Number of transformer layers
	keys      [][]*tensor.Tensor   // [layer][batch, num_heads, seq_len, head_dim]
	values    [][]*tensor.Tensor   // [layer][batch, num_heads, seq_len, head_dim]
	seqLen    int                  // Current sequence length in cache
	maxSeqLen int                  // Maximum supported sequence length
	numHeads  int                  // Number of attention heads
	headDim   int                  // Dimension of each head
	batchSize int                  // Batch size
}

// NewKVCache creates a new KV-cache for autoregressive generation
func NewKVCache(numLayers, batchSize, maxSeqLen, numHeads, headDim int) *KVCache {
	cache := &KVCache{
		numLayers: numLayers,
		keys:      make([][]*tensor.Tensor, numLayers),
		values:    make([][]*tensor.Tensor, numLayers),
		seqLen:    0,
		maxSeqLen: maxSeqLen,
		numHeads:  numHeads,
		headDim:   headDim,
		batchSize: batchSize,
	}

	// Pre-allocate tensors for each layer
	for i := 0; i < numLayers; i++ {
		cache.keys[i] = make([]*tensor.Tensor, batchSize)
		cache.values[i] = make([]*tensor.Tensor, batchSize)

		// Initialize empty tensors (will be allocated on first use)
		for b := 0; b < batchSize; b++ {
			cache.keys[i][b] = nil
			cache.values[i][b] = nil
		}
	}

	return cache
}

// Update updates the cache for a specific layer with new key and value tensors
// k, v: [batch, num_heads, seq_len, head_dim]
func (c *KVCache) Update(layer int, k, v *tensor.Tensor) error {
	if layer < 0 || layer >= c.numLayers {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, c.numLayers)
	}

	kShape := k.Shape()
	vShape := v.Shape()

	// Validate shapes
	if len(kShape) != 4 || len(vShape) != 4 {
		return fmt.Errorf("k and v must be 4D tensors, got shapes %v and %v", kShape, vShape)
	}

	batchSize := kShape[0]

	// For each batch item, concatenate with existing cache
	for b := 0; b < batchSize; b++ {
		if c.keys[layer][b] == nil {
			// First time: allocate and store
			c.keys[layer][b] = extractBatch(k, b)
			c.values[layer][b] = extractBatch(v, b)
		} else {
			// Concatenate along sequence dimension
			c.keys[layer][b] = concatSeq(c.keys[layer][b], extractBatch(k, b))
			c.values[layer][b] = concatSeq(c.values[layer][b], extractBatch(v, b))
		}
	}

	// Update sequence length (use max across all layers)
	if c.keys[layer][0] != nil {
		currentLen := c.keys[layer][0].Shape()[1] // seq dimension
		if currentLen > c.seqLen {
			c.seqLen = currentLen
		}
	}

	return nil
}

// Get retrieves cached keys and values for a specific layer
// Returns: k, v with shapes [batch, num_heads, cached_seq_len, head_dim]
func (c *KVCache) Get(layer int) (k, v *tensor.Tensor, err error) {
	if layer < 0 || layer >= c.numLayers {
		return nil, nil, fmt.Errorf("layer index %d out of range [0, %d)", layer, c.numLayers)
	}

	// If cache is empty for this layer, return nil
	if c.keys[layer][0] == nil {
		return nil, nil, nil
	}

	// Combine all batches into a single tensor
	k = stackBatches(c.keys[layer])
	v = stackBatches(c.values[layer])

	return k, v, nil
}

// Clear resets the cache to empty state
func (c *KVCache) Clear() {
	for i := 0; i < c.numLayers; i++ {
		for b := 0; b < c.batchSize; b++ {
			c.keys[i][b] = nil
			c.values[i][b] = nil
		}
	}
	c.seqLen = 0
}

// Resize changes the maximum sequence length (e.g., for different context windows)
func (c *KVCache) Resize(newMaxSeqLen int) {
	c.maxSeqLen = newMaxSeqLen
	// Note: Actual tensor reallocation happens lazily on next Update
}

// SeqLen returns the current sequence length in the cache
func (c *KVCache) SeqLen() int {
	return c.seqLen
}

// MaxSeqLen returns the maximum supported sequence length
func (c *KVCache) MaxSeqLen() int {
	return c.maxSeqLen
}

// Helper: extract a single batch item from [batch, heads, seq, dim] -> [heads, seq, dim]
func extractBatch(t *tensor.Tensor, batch int) *tensor.Tensor {
	shape := t.Shape()
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	result := tensor.NewTensor([]int{numHeads, seqLen, headDim}, t.DType())

	for h := 0; h < numHeads; h++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < headDim; d++ {
				val := t.At(batch, h, s, d)
				result.Set(val, h, s, d)
			}
		}
	}

	return result
}

// Helper: concatenate two tensors along sequence dimension
// a: [heads, seq1, dim], b: [heads, seq2, dim] -> [heads, seq1+seq2, dim]
func concatSeq(a, b *tensor.Tensor) *tensor.Tensor {
	aShape := a.Shape()
	bShape := b.Shape()

	numHeads := aShape[0]
	seq1 := aShape[1]
	seq2 := bShape[1]
	headDim := aShape[2]

	result := tensor.NewTensor([]int{numHeads, seq1 + seq2, headDim}, a.DType())

	// Copy from a
	for h := 0; h < numHeads; h++ {
		for s := 0; s < seq1; s++ {
			for d := 0; d < headDim; d++ {
				val := a.At(h, s, d)
				result.Set(val, h, s, d)
			}
		}
	}

	// Copy from b
	for h := 0; h < numHeads; h++ {
		for s := 0; s < seq2; s++ {
			for d := 0; d < headDim; d++ {
				val := b.At(h, s, d)
				result.Set(val, h, s+seq1, d)
			}
		}
	}

	return result
}

// Helper: stack batch tensors into a single tensor
// [batch] of [heads, seq, dim] -> [batch, heads, seq, dim]
func stackBatches(batches []*tensor.Tensor) *tensor.Tensor {
	if len(batches) == 0 || batches[0] == nil {
		return nil
	}

	batchSize := len(batches)
	shape := batches[0].Shape()
	numHeads := shape[0]
	seqLen := shape[1]
	headDim := shape[2]

	result := tensor.NewTensor([]int{batchSize, numHeads, seqLen, headDim}, batches[0].DType())

	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < headDim; d++ {
					val := batches[b].At(h, s, d)
					result.Set(val, b, h, s, d)
				}
			}
		}
	}

	return result
}
