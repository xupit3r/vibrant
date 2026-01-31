package llm

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestEngineCreation tests that an engine can be created
func TestEngineCreation(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	
	if engine == nil {
		t.Fatal("Engine is nil")
	}
	
	// Verify engine implements the interface
	var _ Engine = engine
}

// TestEngineGenerate tests basic text generation
func TestEngineGenerate(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	result, err := engine.Generate(ctx, "Test prompt", GenerateOptions{
		MaxTokens:   10,
		Temperature: 0.7,
	})
	
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	
	if result == "" {
		t.Error("Generated result is empty")
	}
	
	// Check that result contains expected content (mock or real)
	if len(result) == 0 {
		t.Error("Expected non-empty result")
	}
}

// TestEngineGenerateStream tests streaming generation
func TestEngineGenerateStream(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	stream, err := engine.GenerateStream(ctx, "Test prompt", GenerateOptions{
		MaxTokens:   10,
		Temperature: 0.7,
	})
	
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	
	// Read from stream
	var chunks []string
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	
	if len(chunks) == 0 {
		t.Error("Expected at least one chunk from stream")
	}
}

// TestEngineTokenCount tests token counting
func TestEngineTokenCount(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	tests := []struct {
		name     string
		text     string
		minCount int
	}{
		{
			name:     "simple text",
			text:     "Hello world",
			minCount: 1,
		},
		{
			name:     "longer text",
			text:     "This is a longer piece of text with multiple words",
			minCount: 5,
		},
		{
			name:     "empty text",
			text:     "",
			minCount: 0,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count := engine.TokenCount(tt.text)
			
			if count < tt.minCount {
				t.Errorf("TokenCount(%q) = %d, want >= %d", tt.text, count, tt.minCount)
			}
		})
	}
}

// TestEngineClose tests proper cleanup
func TestEngineClose(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	
	err = engine.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}
}

// TestEngineOptions tests various option configurations
func TestEngineOptions(t *testing.T) {
	tests := []struct {
		name    string
		options LoadOptions
		wantErr bool
	}{
		{
			name: "default options",
			options: LoadOptions{
				ContextSize: 2048,
				Threads:     4,
			},
			wantErr: false,
		},
		{
			name: "large context",
			options: LoadOptions{
				ContextSize: 8192,
				Threads:     8,
			},
			wantErr: false,
		},
		{
			name: "minimal context",
			options: LoadOptions{
				ContextSize: 512,
				Threads:     1,
			},
			wantErr: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine, err := NewLlamaEngine("", tt.options)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("NewLlamaEngine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			
			if !tt.wantErr && engine == nil {
				t.Error("Expected non-nil engine")
			}
			
			if engine != nil {
				engine.Close()
			}
		})
	}
}

// TestBuildTag verifies the correct build tag is active
func TestBuildTag(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Generate a test response
	result, err := engine.Generate(ctx, "test", GenerateOptions{MaxTokens: 10})
	
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	
	// Check if we're using mock or real engine based on response
	// Mock engine has distinctive response pattern
	isMock := strings.Contains(result, "mock") || strings.Contains(result, "CGO_ENABLED")
	
	t.Logf("Build configuration: mock=%v", isMock)
	t.Logf("Sample response: %s", truncateStr(result, 100))
}

// TestDefaultOptions verifies default option functions
func TestDefaultOptions(t *testing.T) {
	loadOpts := DefaultLoadOptions()
	if loadOpts.ContextSize <= 0 {
		t.Error("DefaultLoadOptions should have positive ContextSize")
	}
	if loadOpts.Threads <= 0 {
		t.Error("DefaultLoadOptions should have positive Threads")
	}
	
	genOpts := DefaultGenerateOptions()
	if genOpts.MaxTokens <= 0 {
		t.Error("DefaultGenerateOptions should have positive MaxTokens")
	}
	if genOpts.Temperature < 0 || genOpts.Temperature > 2 {
		t.Error("DefaultGenerateOptions Temperature should be reasonable")
	}
}

// TestEngineContextCancellation tests context cancellation
func TestEngineContextCancellation(t *testing.T) {
	engine, err := NewLlamaEngine("", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	_, err = engine.Generate(ctx, "test", GenerateOptions{MaxTokens: 100})
	
	// Should handle cancellation gracefully
	// (May or may not return error depending on implementation)
	t.Logf("Generate with cancelled context: error=%v", err)
}

// Helper functions
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

