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
// Enable debug logging
inference.DebugInference = true

homeDir, _ := os.UserHomeDir()
modelPath := filepath.Join(homeDir, ".vibrant", "models", "qwen2.5-coder-3b-q4.gguf")

config := &inference.Config{
MaxTokens:   10,
Temperature: 0.0,
TopP:        1.0,
TopK:        0,
StopTokens:  []int{},
Seed:        42,
Device:      tensor.CPU, // Force CPU
}

fmt.Println("Loading model on CPU...")
engine, err := inference.NewEngine(modelPath, config)
if err != nil {
fmt.Printf("Failed: %v\n", err)
os.Exit(1)
}

prompt := "Hello world"
fmt.Printf("\nPrompt: %q\n\n", prompt)

ctx := context.Background()
opts := inference.GenerateOptions{MaxTokens: 10}

output, err := engine.Generate(ctx, prompt, opts)
if err != nil {
fmt.Printf("Generation failed: %v\n", err)
os.Exit(1)
}

fmt.Printf("\nFinal output: %q\n", output)
}
