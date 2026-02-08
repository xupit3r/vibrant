package main

import (
"context"
"fmt"
"os"
"path/filepath"

"github.com/xupit3r/vibrant/internal/inference"
"github.com/xupit3r/vibrant/internal/tensor"
)

func main() {
inference.DebugInference = true

homeDir, _ := os.UserHomeDir()
modelPath := filepath.Join(homeDir, ".vibrant", "models", "qwen2.5-coder-3b-q4.gguf")

config := &inference.Config{
MaxTokens:   3,
Temperature: 0.0,
Device:      tensor.CPU,
}

engine, _ := inference.NewEngine(modelPath, config)

// Test 1: Simple continuation
fmt.Println("\n=== Test 1: '1 + 1 =' ===")
output, _ := engine.Generate(context.Background(), "1 + 1 =", inference.GenerateOptions{MaxTokens: 3})
fmt.Printf("Output: %q\n", output)

// Test 2: Code completion  
fmt.Println("\n=== Test 2: 'def hello():' ===")
output, _ = engine.Generate(context.Background(), "def hello():", inference.GenerateOptions{MaxTokens: 3})
fmt.Printf("Output: %q\n", output)
}
