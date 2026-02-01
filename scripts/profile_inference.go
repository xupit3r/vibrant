package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/xupit3r/vibrant/internal/inference"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	cpuprofile := flag.String("cpuprofile", "cpu.prof", "Write CPU profile to file")
	memprofile := flag.String("memprofile", "mem.prof", "Write memory profile to file")
	prompt := flag.String("prompt", "Write a function in Go that", "Prompt to use for generation")
	tokens := flag.Int("tokens", 50, "Number of tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		// Try to find a model
		possiblePaths := []string{
			os.Getenv("VIBRANT_TEST_MODEL"),
			os.ExpandEnv("$HOME/.cache/vibrant/models/qwen2.5-coder-14b-q5.gguf"),
			os.ExpandEnv("$HOME/.cache/vibrant/models/qwen2.5-coder-7b-q5.gguf"),
			os.ExpandEnv("$HOME/.cache/vibrant/models/qwen2.5-coder-3b-q5.gguf"),
		}

		for _, path := range possiblePaths {
			if path != "" {
				if _, err := os.Stat(path); err == nil {
					*modelPath = path
					break
				}
			}
		}

		if *modelPath == "" {
			log.Fatal("No model file specified. Use -model flag or set VIBRANT_TEST_MODEL environment variable")
		}
	}

	fmt.Printf("🔍 Profiling inference with model: %s\n", *modelPath)
	fmt.Printf("📝 Prompt: %s\n", *prompt)
	fmt.Printf("🎯 Tokens to generate: %d\n\n", *tokens)

	// Start CPU profiling
	f, err := os.Create(*cpuprofile)
	if err != nil {
		log.Fatal("Could not create CPU profile: ", err)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("Could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()

	// Create inference engine
	config := &inference.Config{
		MaxTokens:   *tokens,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{},
		Seed:        42,
	}

	fmt.Println("⏳ Loading model...")
	startLoad := time.Now()
	engine, err := inference.NewEngine(*modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	loadDuration := time.Since(startLoad)
	fmt.Printf("✅ Model loaded in %v\n\n", loadDuration)

	// Run generation with timing
	ctx := context.Background()
	opts := inference.GenerateOptions{
		MaxTokens:  *tokens,
		StopTokens: []int{},
	}

	fmt.Println("🚀 Starting generation...")
	startGen := time.Now()

	result, err := engine.Generate(ctx, *prompt, opts)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	genDuration := time.Since(startGen)
	tokensGenerated := engine.TokenCount(result)

	fmt.Printf("\n✅ Generation complete!\n\n")
	fmt.Printf("📊 Performance Metrics:\n")
	fmt.Printf("  • Time: %v\n", genDuration)
	fmt.Printf("  • Tokens: %d\n", tokensGenerated)
	fmt.Printf("  • Tokens/sec: %.2f\n", float64(tokensGenerated)/genDuration.Seconds())
	fmt.Printf("  • Time per token: %v\n\n", genDuration/time.Duration(tokensGenerated))

	fmt.Printf("📝 Generated text:\n%s\n\n", result)

	// Write memory profile
	mf, err := os.Create(*memprofile)
	if err != nil {
		log.Fatal("Could not create memory profile: ", err)
	}
	defer mf.Close()

	runtime.GC() // get up-to-date statistics
	if err := pprof.WriteHeapProfile(mf); err != nil {
		log.Fatal("Could not write memory profile: ", err)
	}

	fmt.Printf("💾 Profiles written to:\n")
	fmt.Printf("  • CPU: %s\n", *cpuprofile)
	fmt.Printf("  • Memory: %s\n", *memprofile)
	fmt.Printf("\n📈 Analyze with:\n")
	fmt.Printf("  go tool pprof -http=:8080 %s\n", *cpuprofile)
	fmt.Printf("  go tool pprof -http=:8081 %s\n", *memprofile)
}
