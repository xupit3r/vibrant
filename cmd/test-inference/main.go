package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/xupit3r/vibrant/internal/inference"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Write a function that", "Prompt to test")
	maxTokens := flag.Int("max-tokens", 50, "Maximum tokens to generate")
	stream := flag.Bool("stream", false, "Use streaming generation")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide a model path with -model flag")
	}

	fmt.Printf("═══════════════════════════════════════════════════════════\n")
	fmt.Printf("  Vibrant Custom Inference Engine - End-to-End Test\n")
	fmt.Printf("═══════════════════════════════════════════════════════════\n\n")

	// Test 1: Model Loading
	fmt.Printf("Test 1: Loading GGUF Model\n")
	fmt.Printf("  Model: %s\n", *modelPath)

	startLoad := time.Now()
	config := &inference.Config{
		MaxTokens:   *maxTokens,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{},
		Seed:        42,
	}

	engine, err := inference.NewEngine(*modelPath, config)
	if err != nil {
		log.Fatalf("❌ Failed to create engine: %v", err)
	}
	defer engine.Close()

	loadTime := time.Since(startLoad)
	fmt.Printf("  ✅ Model loaded in %v\n\n", loadTime)

	// Test 2: Token Counting
	fmt.Printf("Test 2: Token Counting\n")
	fmt.Printf("  Text: %q\n", *prompt)

	tokenCount := engine.TokenCount(*prompt)
	fmt.Printf("  ✅ Token count: %d tokens\n\n", tokenCount)

	// Test 3: Text Generation
	ctx := context.Background()

	if *stream {
		// Test 3a: Streaming Generation
		fmt.Printf("Test 3: Streaming Generation\n")
		fmt.Printf("  Prompt: %q\n", *prompt)
		fmt.Printf("  Max tokens: %d\n", *maxTokens)
		fmt.Printf("  Temperature: %.2f\n\n", config.Temperature)

		fmt.Printf("  Response:\n  ──────────────────────────────────────────────────────\n  ")

		startGen := time.Now()
		tokenChan, err := engine.GenerateStream(ctx, *prompt, inference.GenerateOptions{
			MaxTokens:  *maxTokens,
			StopTokens: []int{},
		})
		if err != nil {
			log.Fatalf("❌ Failed to start streaming: %v", err)
		}

		chunkCount := 0
		totalChars := 0
		for chunk := range tokenChan {
			fmt.Print(chunk)
			chunkCount++
			totalChars += len(chunk)
		}
		genTime := time.Since(startGen)

		fmt.Printf("\n  ──────────────────────────────────────────────────────\n")
		fmt.Printf("  ✅ Generated %d chunks (%d chars) in %v\n", chunkCount, totalChars, genTime)

		if genTime.Seconds() > 0 {
			tokensPerSec := float64(*maxTokens) / genTime.Seconds()
			fmt.Printf("  ⚡ Speed: %.2f tokens/sec\n\n", tokensPerSec)
		}
	} else {
		// Test 3b: Blocking Generation
		fmt.Printf("Test 3: Blocking Generation\n")
		fmt.Printf("  Prompt: %q\n", *prompt)
		fmt.Printf("  Max tokens: %d\n", *maxTokens)
		fmt.Printf("  Temperature: %.2f\n\n", config.Temperature)

		startGen := time.Now()
		result, err := engine.Generate(ctx, *prompt, inference.GenerateOptions{
			MaxTokens:  *maxTokens,
			StopTokens: []int{},
		})
		if err != nil {
			log.Fatalf("❌ Generation failed: %v", err)
		}
		genTime := time.Since(startGen)

		fmt.Printf("  Response:\n  ──────────────────────────────────────────────────────\n")
		fmt.Printf("  %s\n", result)
		fmt.Printf("  ──────────────────────────────────────────────────────\n")
		fmt.Printf("  ✅ Generated %d chars in %v\n", len(result), genTime)

		if genTime.Seconds() > 0 {
			tokensPerSec := float64(*maxTokens) / genTime.Seconds()
			fmt.Printf("  ⚡ Speed: %.2f tokens/sec\n\n", tokensPerSec)
		}
	}

	// Test 4: Multiple Generations (cache efficiency)
	fmt.Printf("Test 4: Multiple Generations (testing cache)\n")

	prompts := []string{
		"Hello",
		"What is Go?",
		"Explain recursion",
	}

	for i, p := range prompts {
		fmt.Printf("  Generation %d: %q\n", i+1, p)
		startGen := time.Now()

		result, err := engine.Generate(ctx, p, inference.GenerateOptions{
			MaxTokens:  20,
			StopTokens: []int{},
		})
		if err != nil {
			fmt.Printf("    ❌ Failed: %v\n", err)
			continue
		}

		genTime := time.Since(startGen)
		fmt.Printf("    ✅ %q (%v)\n", truncate(result, 50), genTime)
	}

	// Summary
	fmt.Printf("\n═══════════════════════════════════════════════════════════\n")
	fmt.Printf("  All Tests Completed Successfully! ✅\n")
	fmt.Printf("═══════════════════════════════════════════════════════════\n")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
