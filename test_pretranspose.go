package main

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
	"github.com/xupit3r/vibrant/internal/transformer"
)

func main() {
	modelPath := os.Getenv("HOME") + "/.vibrant/models/qwen2.5-coder-7b-q4.gguf"

	fmt.Println("Loading model...")

	// Increase cache budget to 16GB to reduce eviction
	tensor.DefaultWeightCache.SetBudget(16 * 1024 * 1024 * 1024)

	ggufFile, err := gguf.ParseGGUF(modelPath)
	if err != nil {
		panic(err)
	}

	model, err := transformer.NewModel(ggufFile)
	if err != nil {
		panic(err)
	}

	tokens := [][]int{{1, 100, 200}} // BOS + 2 tokens

	fmt.Println("\n=== First Forward Pass (cache cold) ===")
	runtime.GC()
	start := time.Now()
	_, err = model.Forward(tokens, true)
	if err != nil {
		panic(err)
	}
	duration1 := time.Since(start)
	used1, budget, entries := tensor.DefaultWeightCache.Stats()
	fmt.Printf("Time: %v\n", duration1)
	fmt.Printf("Cache: %d MB used / %d MB budget, %d entries\n\n",
		used1/(1024*1024), budget/(1024*1024), entries)

	fmt.Println("=== Second Forward Pass (cache warm) ===")
	runtime.GC()
	start = time.Now()
	_, err = model.Forward(tokens, true)
	if err != nil {
		panic(err)
	}
	duration2 := time.Since(start)
	used2, budget, entries := tensor.DefaultWeightCache.Stats()
	fmt.Printf("Time: %v\n", duration2)
	fmt.Printf("Cache: %d MB used / %d MB budget, %d entries\n\n",
		used2/(1024*1024), budget/(1024*1024), entries)

	fmt.Println("=== Third Forward Pass (verify cache) ===")
	runtime.GC()
	start = time.Now()
	_, err = model.Forward(tokens, true)
	if err != nil {
		panic(err)
	}
	duration3 := time.Since(start)
	used3, budget, entries := tensor.DefaultWeightCache.Stats()
	fmt.Printf("Time: %v\n", duration3)
	fmt.Printf("Cache: %d MB used / %d MB budget, %d entries\n\n",
		used3/(1024*1024), budget/(1024*1024), entries)

	fmt.Println("=== Performance Summary ===")
	fmt.Printf("Pass 1 (cold):  %v\n", duration1)
	fmt.Printf("Pass 2 (warm):  %v\n", duration2)
	fmt.Printf("Pass 3 (warm):  %v\n", duration3)
	if duration1 > duration2 {
		speedup := float64(duration1) / float64(duration2)
		fmt.Printf("Speedup (cold->warm): %.2fx\n", speedup)
	}
}
