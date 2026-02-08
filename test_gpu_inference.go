package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/xupit3r/vibrant/internal/inference"
	"github.com/xupit3r/vibrant/internal/tensor"
)

func main() {
	// Enable debug logging
	inference.DebugInference = true
	
	fmt.Println("=== GPU Inference Test ===")
	fmt.Println()

	// Find model file
	homeDir, err := os.UserHomeDir()
	if err != nil {
		fmt.Printf("Error getting home dir: %v\n", err)
		os.Exit(1)
	}

	modelPath := filepath.Join(homeDir, ".vibrant", "models", "qwen2.5-coder-3b-q4.gguf")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Model not found: %s\n", modelPath)
		fmt.Println("Please download the model first")
		os.Exit(1)
	}

	fmt.Printf("Model: %s\n", modelPath)
	fmt.Println()

	fmt.Println("--- Test 1: GPU Inference ---")
	fmt.Println("NOTE: Running with DEBUG mode enabled")
	if err := testGPUInference(modelPath); err != nil {
		fmt.Printf("GPU test FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("GPU test PASSED ✓")
	fmt.Println()

	// Test 2: CPU Inference (baseline)
	fmt.Println("--- Test 2: CPU Inference (baseline) ---")
	fmt.Println("NOTE: Running with DEBUG mode enabled")
	if err := testCPUInference(modelPath); err != nil {
		fmt.Printf("CPU test FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("CPU test PASSED ✓")
	fmt.Println()

	fmt.Println("=== All Tests Passed ===")
}

func testGPUInference(modelPath string) error {
	config := &inference.Config{
		MaxTokens:   5, // Very short for debugging
		Temperature: 0.0, // Greedy for determinism
		TopP:        1.0,
		TopK:        0,
		StopTokens:  []int{},
		Seed:        42,
		Device:      tensor.GPU,
	}

	fmt.Println("Loading model on GPU...")
	start := time.Now()
	engine, err := inference.NewEngine(modelPath, config)
	if err != nil {
		return fmt.Errorf("failed to create engine: %w", err)
	}
	loadTime := time.Since(start)
	fmt.Printf("Model loaded in %.2fs\n", loadTime.Seconds())

	// Simple test prompt
	prompt := "Hello"
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Println("Generating (5 tokens max)...")

	ctx := context.Background()
	opts := inference.GenerateOptions{
		MaxTokens: 5,
	}

	start = time.Now()
	output, err := engine.Generate(ctx, prompt, opts)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}
	genTime := time.Since(start)

	fmt.Printf("Output: %q\n", output)
	fmt.Printf("Generation time: %.2fs\n", genTime.Seconds())

	// Validate output
	if len(output) == 0 {
		return fmt.Errorf("empty output (no output bug)")
	}
	if output == prompt {
		return fmt.Errorf("output equals prompt (no generation)")
	}

	return nil
}

func testCPUInference(modelPath string) error {
	config := &inference.Config{
		MaxTokens:   5, // Very short for debugging
		Temperature: 0.0, // Greedy for determinism
		TopP:        1.0,
		TopK:        0,
		StopTokens:  []int{},
		Seed:        42,
		Device:      tensor.CPU,
	}

	fmt.Println("Loading model on CPU...")
	start := time.Now()
	engine, err := inference.NewEngine(modelPath, config)
	if err != nil {
		return fmt.Errorf("failed to create engine: %w", err)
	}
	loadTime := time.Since(start)
	fmt.Printf("Model loaded in %.2fs\n", loadTime.Seconds())

	prompt := "Write a hello world program in Python:\n\n"
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Println("Generating (5 tokens max)...")

	ctx := context.Background()
	opts := inference.GenerateOptions{
		MaxTokens: 5,
	}

	start = time.Now()
	output, err := engine.Generate(ctx, prompt, opts)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}
	genTime := time.Since(start)

	fmt.Printf("Output: %q\n", output)
	fmt.Printf("Generation time: %.2fs\n", genTime.Seconds())

	// Validate output
	if len(output) == 0 {
		return fmt.Errorf("empty output")
	}

	return nil
}
