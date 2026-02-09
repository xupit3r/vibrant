package tokenizer

import (
	"strings"
	"unicode/utf8"
)

// Encode converts text to token IDs using BPE algorithm
// If addBOS is true, prepends BOS token
// If addEOS is true, appends EOS token
func (t *Tokenizer) Encode(text string, addBOS, addEOS bool) []int {
	if text == "" {
		result := make([]int, 0)
		if addBOS && t.bosID >= 0 {
			result = append(result, t.bosID)
		}
		if addEOS && t.eosID >= 0 {
			result = append(result, t.eosID)
		}
		return result
	}

	// GPT2-style preprocessing: replace spaces with Ġ (U+0120)
	// This is required for Qwen and other GPT2-based tokenizers
	if t.modelType == "gpt2" {
		text = strings.ReplaceAll(text, " ", "Ġ")
	}

	// For GPT2 tokenizers, split into UTF-8 characters (runes)
	// For byte-level BPE, each character is a token initially
	var tokens []string
	if t.modelType == "gpt2" {
		// Split into individual runes (characters), not bytes
		for _, r := range text {
			tokens = append(tokens, string(r))
		}
	} else {
		// Original byte-level BPE
		bytes := []byte(text)
		tokens = make([]string, len(bytes))
		for i, b := range bytes {
			tokens[i] = string([]byte{b})
		}
	}

	// Apply BPE merges iteratively
	for {
		// Find the highest-priority merge that exists in current tokens
		bestPair := ""
		bestRank := -1
		bestPos := -1

		for i := 0; i < len(tokens)-1; i++ {
			pair := tokens[i] + " " + tokens[i+1]
			if rank, ok := t.merges[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestPair = pair
					bestPos = i
				}
			}
		}

		// No more merges possible
		if bestRank == -1 {
			break
		}

		// Apply the merge
		parts := strings.Split(bestPair, " ")
		merged := parts[0] + parts[1]

		// Replace the pair at bestPos with the merged token
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestPos]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestPos+2:]...)
		tokens = newTokens
	}

	// Convert token strings to IDs
	ids := make([]int, 0, len(tokens)+2)

	if addBOS && t.bosID >= 0 {
		ids = append(ids, t.bosID)
	}

	for _, token := range tokens {
		if id, ok := t.vocab[token]; ok {
			ids = append(ids, id)
		} else {
			// Unknown token
			if t.unkID >= 0 {
				ids = append(ids, t.unkID)
			}
		}
	}

	if addEOS && t.eosID >= 0 {
		ids = append(ids, t.eosID)
	}

	return ids
}

// Decode converts token IDs back to text
// If skipSpecial is true, skips BOS/EOS/PAD tokens
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string {
	if len(ids) == 0 {
		return ""
	}

	// Convert IDs to token strings
	var builder strings.Builder

	for _, id := range ids {
		// Skip special tokens if requested
		if skipSpecial {
			if id == t.bosID || id == t.eosID || id == t.padID {
				continue
			}
		}

		// Get token string
		token := t.IDToToken(id)
		if token == "" {
			continue // Skip invalid IDs
		}

		builder.WriteString(token)
	}

	return builder.String()
}

// EncodeAsTokens converts text to token strings (for debugging)
func (t *Tokenizer) EncodeAsTokens(text string) []string {
	ids := t.Encode(text, false, false)
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = t.IDToToken(id)
	}
	return tokens
}

// DecodeTokens converts token strings to text (for debugging)
func (t *Tokenizer) DecodeTokens(tokens []string) string {
	var builder strings.Builder
	for _, token := range tokens {
		builder.WriteString(token)
	}
	return builder.String()
}

// CountTokens returns the number of tokens in text (without special tokens)
func (t *Tokenizer) CountTokens(text string) int {
	return len(t.Encode(text, false, false))
}

// IsValidUTF8 checks if a byte sequence is valid UTF-8
func isValidUTF8(b []byte) bool {
	return utf8.Valid(b)
}
