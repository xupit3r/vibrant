package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	tensorName := flag.String("tensor", "token_embd.weight", "Tensor name to load")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide a model path with -model flag")
	}

	fmt.Printf("Loading GGUF file: %s\n", *modelPath)

	file, err := gguf.ParseGGUF(*modelPath)
	if err != nil {
		log.Fatalf("Failed to parse GGUF: %v", err)
	}

	fmt.Printf("GGUF file parsed successfully\n")
	fmt.Printf("Total tensors: %d\n\n", len(file.Tensors))

	// Check if tensor exists
	fmt.Printf("Looking for tensor: %s\n", *tensorName)
	info, ok := file.Tensors[*tensorName]
	if !ok {
		fmt.Printf("❌ Tensor not found in map\n")
		fmt.Printf("\nAvailable tensors:\n")
		for name := range file.Tensors {
			fmt.Printf("  - %s\n", name)
			if len(file.Tensors) > 20 {
				fmt.Printf("  ... (%d more)\n", len(file.Tensors)-1)
				break
			}
		}
		return
	}

	fmt.Printf("✅ Tensor found in map\n")
	fmt.Printf("  Dims: %v\n", info.Dims)
	fmt.Printf("  Type: %d\n", info.Type)
	fmt.Printf("  Offset: %d\n", info.Offset)
	fmt.Printf("  Size: %d bytes\n\n", info.Size)

	// Try to load the tensor
	fmt.Printf("Attempting to load tensor...\n")
	t, err := file.LoadTensor(*tensorName)
	if err != nil {
		log.Fatalf("❌ Failed to load tensor: %v", err)
	}

	fmt.Printf("✅ Tensor loaded successfully!\n")
	fmt.Printf("  Shape: %v\n", t.Shape())
	fmt.Printf("  DType: %v\n", t.DType())
}
