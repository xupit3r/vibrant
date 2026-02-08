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
// Enable debug
inference.DebugInference = true

homeDir, _ := os.UserHomeDir()
modelPath := filepath.Join(homeDir, ".vibrant", "models", "qwen2.5-coder-3b-q4.gguf")

config := &inference.Config{
MaxTokens:   5,
Temperature: 0.0,
Device:      tensor.CPU,
}

fmt.Println("Loading 3B model...")
engine, err := inference.NewEngine(modelPath, config)
if err != nil {
fmt.Printf("Failed: %v\n", err)
os.Exit(1)
}

prompt := "The"
fmt.Printf("\nPrompt: %q\n\n", prompt)

opts := inference.GenerateOptions{MaxTokens: 5}
output, err := engine.Generate(context.Background(), prompt, opts)
if err != nil {
fmt.Printf("Failed: %v\n", err)
os.Exit(1)
}

fmt.Printf("\nFinal output: %q\n", output)
}
