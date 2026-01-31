package transformer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// Embeddings layer converts token IDs to continuous vectors
type Embeddings struct {
	// Weight matrix: [vocab_size, hidden_dim]
	weight *tensor.Tensor

	vocabSize int
	hiddenDim int
}

// NewEmbeddings creates an embeddings layer from GGUF weights
func NewEmbeddings(ggufFile *gguf.GGUFFile, cfg *Config) (*Embeddings, error) {
	// Load embedding weight tensor
	// Common names: "token_embd.weight", "model.embed_tokens.weight"
	var weight *tensor.Tensor
	var err error

	// Try common embedding tensor names
	tensorNames := []string{
		"token_embd.weight",
		"model.embed_tokens.weight",
		"tok_embeddings.weight",
	}

	for _, name := range tensorNames {
		weight, err = ggufFile.LoadTensor(name)
		if err == nil {
			break
		}
	}

	if weight == nil {
		return nil, fmt.Errorf("embedding weight tensor not found (tried: %v)", tensorNames)
	}

	// Verify shape: GGUF stores as [hidden_dim, vocab_size] (transposed)
	shape := weight.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("embedding weight must be 2D, got shape %v", shape)
	}

	// Check if dimensions match (either order)
	if (shape[0] == cfg.VocabSize && shape[1] == cfg.HiddenDim) ||
		(shape[0] == cfg.HiddenDim && shape[1] == cfg.VocabSize) {
		// Dimensions are correct, we'll handle the transpose during lookup
	} else {
		return nil, fmt.Errorf("embedding shape mismatch: weight has %v, expected [%d, %d] or [%d, %d]",
			shape, cfg.VocabSize, cfg.HiddenDim, cfg.HiddenDim, cfg.VocabSize)
	}

	return &Embeddings{
		weight:    weight,
		vocabSize: cfg.VocabSize,
		hiddenDim: cfg.HiddenDim,
	}, nil
}

// Forward performs embedding lookup
// Input: token IDs [batch_size, seq_len]
// Output: embeddings [batch_size, seq_len, hidden_dim]
func (e *Embeddings) Forward(tokenIDs [][]int) (*tensor.Tensor, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("empty token IDs")
	}

	batchSize := len(tokenIDs)
	seqLen := len(tokenIDs[0])

	// Validate all sequences have same length
	for i, seq := range tokenIDs {
		if len(seq) != seqLen {
			return nil, fmt.Errorf("sequence %d has length %d, expected %d", i, len(seq), seqLen)
		}
	}

	// Validate token IDs are in range
	for i, seq := range tokenIDs {
		for j, id := range seq {
			if id < 0 || id >= e.vocabSize {
				return nil, fmt.Errorf("token ID at [%d,%d] is %d, out of range [0,%d)",
					i, j, id, e.vocabSize)
			}
		}
	}

	// Create output tensor: [batch_size, seq_len, hidden_dim]
	output := tensor.NewTensor([]int{batchSize, seqLen, e.hiddenDim}, tensor.Float32)

	// Look up embeddings
	// For each token ID, copy the corresponding row/column from the weight matrix
	// GGUF stores embeddings as [hidden_dim, vocab_size] so we need to access column for each token
	weightShape := e.weight.Shape()
	transposed := weightShape[0] == e.hiddenDim // true if [hidden_dim, vocab_size]

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			tokenID := tokenIDs[b][s]

			// Get embedding vector for this token
			for d := 0; d < e.hiddenDim; d++ {
				var val float32
				if transposed {
					// Weight is [hidden_dim, vocab_size], access weight[d, tokenID]
					val = e.weight.At(d, tokenID)
				} else {
					// Weight is [vocab_size, hidden_dim], access weight[tokenID, d]
					val = e.weight.At(tokenID, d)
				}
				output.Set(val, b, s, d)
			}
		}
	}

	return output, nil
}

// Embed is a convenience method for single sequence
// Input: token IDs [seq_len]
// Output: embeddings [1, seq_len, hidden_dim]
func (e *Embeddings) Embed(tokenIDs []int) (*tensor.Tensor, error) {
	return e.Forward([][]int{tokenIDs})
}
