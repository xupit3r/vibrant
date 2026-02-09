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
	oData := output.Data().([]float32)

	// Look up embeddings
	weightShape := e.weight.Shape()
	transposed := weightShape[0] == e.hiddenDim // true if [hidden_dim, vocab_size]

	// Debug: log token IDs and whether we're transposed
	fmt.Printf("[EMBED] Tokens: %v, transposed=%v, dtype=%v\n", tokenIDs, transposed, e.weight.DType())

	// For quantized embeddings, we need At() -- but for Float32 we can use direct access
	if e.weight.DType() == tensor.Float32 {
		wData := e.weight.Data().([]float32)
		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				tokenID := tokenIDs[b][s]
				outOff := (b*seqLen + s) * e.hiddenDim

				if transposed {
					// Weight is [hidden_dim, vocab_size] -- need to gather column
					for d := 0; d < e.hiddenDim; d++ {
						oData[outOff+d] = wData[d*e.vocabSize+tokenID]
					}
				} else {
					// Weight is [vocab_size, hidden_dim] -- copy row directly
					wOff := tokenID * e.hiddenDim
					copy(oData[outOff:outOff+e.hiddenDim], wData[wOff:wOff+e.hiddenDim])
				}
			}
		}
	} else {
		// Fallback for quantized embeddings -- use At()
		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				tokenID := tokenIDs[b][s]
				outOff := (b*seqLen + s) * e.hiddenDim

				for d := 0; d < e.hiddenDim; d++ {
					if transposed {
						oData[outOff+d] = e.weight.At(d, tokenID)
					} else {
						oData[outOff+d] = e.weight.At(tokenID, d)
					}
				}
			}
		}
	}

	// Debug: show statistics for first embedding vector
	if batchSize > 0 && seqLen > 0 {
		firstEmbedding := oData[0:e.hiddenDim]
		sum, min, max := float32(0), firstEmbedding[0], firstEmbedding[0]
		for _, v := range firstEmbedding {
			sum += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		mean := sum / float32(e.hiddenDim)
		fmt.Printf("[EMBED] First token embedding stats: min=%.4f, max=%.4f, mean=%.4f\n", min, max, mean)
	}

	return output, nil
}

// MoveToDevice moves embeddings to the specified device
// Note: Embeddings stay on CPU because lookup is more efficient there
func (e *Embeddings) MoveToDevice(device tensor.Device) error {
	// Embeddings perform lookups, not matmul, so keep them on CPU
	// GPU lookups would be slower due to random access patterns
	return nil
}

// Embed is a convenience method for single sequence
// Input: token IDs [seq_len]
// Output: embeddings [1, seq_len, hidden_dim]
func (e *Embeddings) Embed(tokenIDs []int) (*tensor.Tensor, error) {
	return e.Forward([][]int{tokenIDs})
}
