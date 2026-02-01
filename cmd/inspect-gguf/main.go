package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/xupit3r/vibrant/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	tensorName := flag.String("tensor", "", "Tensor name to inspect (default: list all)")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide -model path")
	}

	// Load GGUF file
	gf, err := gguf.ParseGGUF(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load GGUF file: %v", err)
	}

	if *tensorName != "" {
		// Inspect specific tensor
		info, err := gf.GetTensorInfo(*tensorName)
		if err != nil {
			log.Fatalf("Failed to get tensor info: %v", err)
		}

		fmt.Printf("Tensor: %s\n", *tensorName)
		fmt.Printf("  Type: %s (GGML code: %d)\n", info.Type, info.Type)
		fmt.Printf("  Shape: %v\n", info.Dims)
		fmt.Printf("  Size: %d bytes\n", info.Size)
		fmt.Printf("  Offset: %d\n", info.Offset)

		// Calculate expected size for different types
		elements := uint64(1)
		for _, dim := range info.Dims {
			elements *= dim
		}

		fmt.Printf("\nCalculated sizes for %d elements:\n", elements)
		fmt.Printf("  Float32 (4 bytes/elem):  %d bytes\n", elements*4)
		fmt.Printf("  Float16 (2 bytes/elem):  %d bytes\n", elements*2)
		fmt.Printf("  Q5_K (176 bytes/256):    %d bytes\n", ((elements+255)/256)*176)
		fmt.Printf("  Q4_K (144 bytes/256):    %d bytes\n", ((elements+255)/256)*144)
		fmt.Printf("  Q8_0 (68 bytes/32):      %d bytes\n", ((elements+31)/32)*68)

		fmt.Printf("\nActual size: %d bytes\n", info.Size)
	} else {
		// List all tensors matching attn_v
		fmt.Println("All attn_v tensors:")
		for name, info := range gf.Tensors {
			if len(name) > 6 && name[len(name)-6:] == "_v.weight" {
				elements := uint64(1)
				for _, dim := range info.Dims {
					elements *= dim
				}
				fmt.Printf("%s: type=%d, shape=%v, size=%d bytes, expected_f32=%d, expected_q5k=%d\n",
					name, info.Type, info.Dims, info.Size, elements*4, ((elements+255)/256)*176)
			}
		}
	}
}
