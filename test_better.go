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
inference.DebugInference = false // Disable debug for cleaner output

homeDir, _ := os.UserHomeDir()
modelPath := filepath.Join(homeDir, ".vibrant", "models", "qwen2.5-coder-3b-q4.gguf")

config := &inference.Config{
MaxTokens:   20,
Temperature: 0.8, // Add some randomness
Device:      tensor.CPU,
}

fmt.Println("Loading model...")
engine, err := inference.NewEngine(modelPath, config)
if err != nil {
fmt.Printf("Failed: %v\n", err)
os.Exit(1)
}

prompts := []string{
"Hello, how are you?",
"Write a Python function to add two numbers:",
"The quick brown fox",
}

for _, prompt := range prompts {
fmt.Printf("\nPrompt: %q\n", prompt)
opts := inference.GenerateOptions{MaxTokens: 20}
output, err := engine.Generate(context.Background(), prompt, opts)
if err != nil {
fmt.Printf("  Error: %v\n", err)
continue
}
fmt.Printf("Output: %q\n", output)
}
}
