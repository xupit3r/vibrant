package main

import (
	"fmt"
	"github.com/xupit3r/vibrant/internal/tokenizer"
	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	ggufFile, _ := gguf.ParseGGUF("/Users/joe/.vibrant/models/qwen2.5-coder-3b-q4.gguf")
	tok, _ := tokenizer.NewTokenizerFromGGUF(ggufFile)
	
	fmt.Printf("Vocab size: %d\n", tok.VocabSize())
	fmt.Printf("BOS ID: %d\n", tok.BOSID())
	fmt.Printf("EOS ID: %d\n", tok.EOSID())
	fmt.Printf("UNK ID: %d\n", tok.UNKID())
	fmt.Printf("Model type: %s\n\n", tok.ModelType())
	
	// Check if specific merges exist
	testPairs := []string{
		"H e",
		"e l",
		"l l",
		"l o",
		"w o",
		"o r",
		"r l",
		"l d",
	}
	
	fmt.Println("Checking merges:")
	for _, pair := range testPairs {
		if tok.HasMerge(pair) {
			fmt.Printf("  %-10s: YES (rank %d)\n", pair, tok.GetMergeRank(pair))
		} else {
			fmt.Printf("  %-10s: NO\n", pair)
		}
	}
	
	// Check what tokens exist for space
	fmt.Println("\nTokens containing space:")
	spaceTokens := []string{
		" ",
		"Ġ",  // GPT2-style space prefix
		"▁",  // SentencePiece space prefix
	}
	for _, token := range spaceTokens {
		id := tok.TokenToID(token)
		if id >= 0 {
			fmt.Printf("  %q -> ID %d\n", token, id)
		} else {
			fmt.Printf("  %q -> NOT FOUND\n", token)
		}
	}
}
