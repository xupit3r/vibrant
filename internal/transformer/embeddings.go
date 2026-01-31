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

	// Verify shape: [vocab_size, hidden_dim]
	shape := weight.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("embedding weight must be 2D, got shape %v", shape)
	}
	if shape[0] != cfg.VocabSize {
		return nil, fmt.Errorf("embedding vocab size mismatch: weight has %d, config has %d",
			shape[0], cfg.VocabSize)
	}
	if shape[1] != cfg.HiddenDim {
		return nil, fmt.Errorf("embedding hidden dim mismatch: weight has %d, config has %d",
			shape[1], cfg.HiddenDim)
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
	// For each token ID, copy the corresponding row from the weight matrix
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			tokenID := tokenIDs[b][s]

			// Get embedding vector for this token
			// embedding = weight[tokenID, :]
			for d := 0; d < e.hiddenDim; d++ {
				val := e.weight.At(tokenID, d)
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
