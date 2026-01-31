package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide a model path with -model flag")
	}

	fmt.Printf("Loading GGUF file: %s\n\n", *modelPath)

	file, err := gguf.ParseGGUF(*modelPath)
	if err != nil {
		log.Fatalf("Failed to parse GGUF: %v", err)
	}

	arch := "unknown"
	if v, ok := file.Metadata["general.architecture"]; ok {
		if s, ok := v.(string); ok {
			arch = s
		}
	}
	fmt.Printf("Architecture: %s\n", arch)
	fmt.Printf("Total tensors: %d\n\n", len(file.Tensors))

	fmt.Printf("Tensor Names:\n")
	fmt.Printf("═══════════════════════════════════════════════════════════\n")

	for name := range file.Tensors {
		fmt.Printf("  %s\n", name)
	}

	fmt.Printf("═══════════════════════════════════════════════════════════\n")
	fmt.Printf("Total: %d tensors\n", len(file.Tensors))
}
