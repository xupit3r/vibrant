package main

import (
	"fmt"
	"github.com/xupit3r/vibrant/internal/tokenizer"
	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	ggufFile, _ := gguf.ParseGGUF("/Users/joe/.vibrant/models/qwen2.5-coder-3b-q4.gguf")
	tok, _ := tokenizer.NewTokenizerFromGGUF(ggufFile)
	
	// Test various prompts
	tests := []string{
		"Hello world",
		"Hello",
		" world",
		"world",
		" ",
		"  ",
	}
	
	for _, test := range tests {
		tokens := tok.Encode(test, false, false)
		fmt.Printf("%-15q -> %v\n", test, tokens)
		for i, t := range tokens {
			decoded := tok.Decode([]int{t}, false)
			fmt.Printf("  [%d] Token %d: %q\n", i, t, decoded)
		}
	}
}
