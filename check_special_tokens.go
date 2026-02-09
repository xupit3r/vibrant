package main

import (
	"fmt"
	"github.com/xupit3r/vibrant/internal/tokenizer"
	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	ggufFile, _ := gguf.ParseGGUF("/Users/joe/.vibrant/models/qwen2.5-coder-3b-q4.gguf")
	tok, _ := tokenizer.NewTokenizerFromGGUF(ggufFile)
	
	// Check special tokens around 128000
	fmt.Println("Special tokens in 128xxx range:")
	for i := 128000; i <= 128010; i++ {
		text := tok.Decode([]int{i}, false)
		fmt.Printf("Token %d: %q\n", i, text)
	}
	
	fmt.Println("\nTokens in our prompt:")
	testTokens := []int{151643, 9707, 128244, 14615}
	for _, t := range testTokens {
		fmt.Printf("Token %d: %q\n", t, tok.Decode([]int{t}, false))
	}
}
