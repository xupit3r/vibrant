package tokenizer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
)

// Tokenizer implements BPE (Byte-Pair Encoding) tokenization
type Tokenizer struct {
	// Vocabulary: token string → token ID
	vocab map[string]int

	// Reverse vocabulary: token ID → token string
	tokens []string

	// Token scores (used for ranking in BPE merges)
	scores []float32

	// BPE merge rules: "token1 token2" → merge rank
	// Lower rank = applied earlier (higher priority)
	merges map[string]int

	// Special token IDs
	bosID int // Beginning of sequence
	eosID int // End of sequence
	padID int // Padding token
	unkID int // Unknown token

	// Model type (e.g., "gpt2", "llama")
	modelType string

	// Cached list of special tokens (e.g. "<|im_start|>", "<|eot_id|>")
	// Built lazily on first use by scanning the vocab.
	specialTokens []string
}

// NewTokenizer creates a new empty tokenizer
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		vocab:  make(map[string]int),
		tokens: make([]string, 0),
		scores: make([]float32, 0),
		merges: make(map[string]int),
		bosID:  -1,
		eosID:  -1,
		padID:  -1,
		unkID:  -1,
	}
}

// NewTokenizerFromGGUF creates a tokenizer from GGUF metadata
func NewTokenizerFromGGUF(ggufFile *gguf.GGUFFile) (*Tokenizer, error) {
	t := NewTokenizer()

	// Get tokenizer model type
	if modelType, ok := ggufFile.Metadata[gguf.KeyTokenizerModel].(string); ok {
		t.modelType = modelType
	} else {
		t.modelType = "unknown"
	}

	// Load tokens
	tokens := ggufFile.GetTokens()
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens found in GGUF metadata")
	}
	t.tokens = tokens

	// Build vocab map (token → ID)
	t.vocab = make(map[string]int, len(tokens))
	for id, token := range tokens {
		t.vocab[token] = id
	}

	// Load token scores (optional)
	scores := ggufFile.GetTokenScores()
	if len(scores) == len(tokens) {
		t.scores = scores
	} else {
		// Default scores: 0.0 for all tokens
		t.scores = make([]float32, len(tokens))
	}

	// Load BPE merges
	mergeStrs := ggufFile.GetMerges()
	t.merges = make(map[string]int, len(mergeStrs))
	for rank, merge := range mergeStrs {
		t.merges[merge] = rank
	}

	// Load special token IDs
	if bosID, ok := ggufFile.Metadata[gguf.KeyTokenizerBOSID]; ok {
		if id, ok := convertToInt(bosID); ok {
			t.bosID = id
		}
	}

	if eosID, ok := ggufFile.Metadata[gguf.KeyTokenizerEOSID]; ok {
		if id, ok := convertToInt(eosID); ok {
			t.eosID = id
		}
	}

	if padID, ok := ggufFile.Metadata[gguf.KeyTokenizerPADID]; ok {
		if id, ok := convertToInt(padID); ok {
			t.padID = id
		}
	}

	// UNK token is typically ID 0 for many models, but not always defined
	if unkToken, ok := t.vocab["<unk>"]; ok {
		t.unkID = unkToken
	} else if unkToken, ok := t.vocab["<UNK>"]; ok {
		t.unkID = unkToken
	}

	return t, nil
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.tokens)
}

// BOSID returns the beginning-of-sequence token ID
func (t *Tokenizer) BOSID() int {
	return t.bosID
}

// EOSID returns the end-of-sequence token ID
func (t *Tokenizer) EOSID() int {
	return t.eosID
}

// PADID returns the padding token ID
func (t *Tokenizer) PADID() int {
	return t.padID
}

// UNKID returns the unknown token ID
func (t *Tokenizer) UNKID() int {
	return t.unkID
}

// ModelType returns the tokenizer model type
func (t *Tokenizer) ModelType() string {
	return t.modelType
}

// TokenToID converts a token string to its ID
// Returns -1 if token not found
func (t *Tokenizer) TokenToID(token string) int {
	if id, ok := t.vocab[token]; ok {
		return id
	}
	return -1
}

// IDToToken converts a token ID to its string
// Returns empty string if ID is out of range
func (t *Tokenizer) IDToToken(id int) string {
	if id >= 0 && id < len(t.tokens) {
		return t.tokens[id]
	}
	return ""
}

// HasMerge checks if a merge rule exists for a token pair
func (t *Tokenizer) HasMerge(pair string) bool {
	_, ok := t.merges[pair]
	return ok
}

// GetMergeRank returns the rank of a merge rule (lower = higher priority)
// Returns -1 if merge doesn't exist
func (t *Tokenizer) GetMergeRank(pair string) int {
	if rank, ok := t.merges[pair]; ok {
		return rank
	}
	return -1
}

// convertToInt converts various integer types to int
func convertToInt(val interface{}) (int, bool) {
	switch v := val.(type) {
	case int:
		return v, true
	case int8:
		return int(v), true
	case int16:
		return int(v), true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	case uint:
		return int(v), true
	case uint8:
		return int(v), true
	case uint16:
		return int(v), true
	case uint32:
		return int(v), true
	case uint64:
		return int(v), true
	default:
		return 0, false
	}
}
