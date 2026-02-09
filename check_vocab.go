package main

import (
	"fmt"
	"github.com/xupit3r/vibrant/internal/tokenizer"
	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	ggufFile, _ := gguf.ParseGGUF("/Users/joe/.vibrant/models/qwen2.5-coder-3b-q4.gguf")
	tok, _ := tokenizer.NewTokenizerFromGGUF(ggufFile)
	
	// Check what token "Ġworld" is (space + world)
	testTokens := []string{
		"Ġworld",
		"Ġ",
		"world",
		"Hello",
		"ĠHello",
		"Ġw",
		"Ġwor",
		"Ġworld",
	}
	
	fmt.Println("Checking tokens:")
	for _, token := range testTokens {
		id := tok.TokenToID(token)
		if id >= 0 {
			fmt.Printf("  %-15q -> ID %d\n", token, id)
		} else {
			fmt.Printf("  %-15q -> NOT FOUND\n", token)
		}
	}
}
