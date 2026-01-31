package inference

import (
	"context"
	"testing"
)

// BenchmarkSamplerGreedy benchmarks greedy sampling
func BenchmarkSamplerGreedy(b *testing.B) {
	sampler := NewSampler(0.0, 0.0, 0, 42)

	// Create mock logits tensor (vocab size 50000)
	logits := createMockLogits(50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.Sample(logits)
	}
}

// BenchmarkSamplerTemperature benchmarks temperature sampling
func BenchmarkSamplerTemperature(b *testing.B) {
	sampler := NewSampler(0.7, 0.0, 0, 42)

	logits := createMockLogits(50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.Sample(logits)
	}
}

// BenchmarkSamplerTopK benchmarks top-k sampling
func BenchmarkSamplerTopK(b *testing.B) {
	sampler := NewSampler(0.7, 0.0, 40, 42)

	logits := createMockLogits(50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.Sample(logits)
	}
}

// BenchmarkSamplerTopP benchmarks top-p (nucleus) sampling
func BenchmarkSamplerTopP(b *testing.B) {
	sampler := NewSampler(0.7, 0.95, 0, 42)

	logits := createMockLogits(50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.Sample(logits)
	}
}

// BenchmarkSamplerTopKTopP benchmarks combined top-k and top-p
func BenchmarkSamplerTopKTopP(b *testing.B) {
	sampler := NewSampler(0.7, 0.95, 40, 42)

	logits := createMockLogits(50000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.Sample(logits)
	}
}

// BenchmarkGenerate benchmarks full generation (requires real model)
func BenchmarkGenerate(b *testing.B) {
	// Skip if no test model available
	modelPath := getTestModelPath()
	if modelPath == "" {
		b.Skip("No test model available (set VIBRANT_TEST_MODEL)")
	}

	config := &Config{
		MaxTokens:   50,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{},
		Seed:        42,
	}

	engine, err := NewEngine(modelPath, config)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	ctx := context.Background()
	opts := GenerateOptions{
		MaxTokens:  50,
		StopTokens: []int{},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := engine.Generate(ctx, "Write a function that", opts)
		if err != nil {
			b.Fatalf("Generate failed: %v", err)
		}
	}
}

// BenchmarkGenerateStream benchmarks streaming generation
func BenchmarkGenerateStream(b *testing.B) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		b.Skip("No test model available (set VIBRANT_TEST_MODEL)")
	}

	config := &Config{
		MaxTokens:   50,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{},
		Seed:        42,
	}

	engine, err := NewEngine(modelPath, config)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	ctx := context.Background()
	opts := GenerateOptions{
		MaxTokens:  50,
		StopTokens: []int{},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream, err := engine.GenerateStream(ctx, "Write a function that", opts)
		if err != nil {
			b.Fatalf("GenerateStream failed: %v", err)
		}

		// Consume stream
		for range stream {
		}
	}
}

// BenchmarkTokenCount benchmarks token counting
func BenchmarkTokenCount(b *testing.B) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		b.Skip("No test model available (set VIBRANT_TEST_MODEL)")
	}

	config := &Config{
		MaxTokens:   100,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []int{},
		Seed:        42,
	}

	engine, err := NewEngine(modelPath, config)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	text := "This is a sample text with multiple words to test tokenization performance."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.TokenCount(text)
	}
}

// getTestModelPath returns path to test model from environment
func getTestModelPath() string {
	// Check environment variable
	// For now, return empty to skip benchmarks that need models
	return ""
}
