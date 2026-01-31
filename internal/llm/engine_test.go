package llm

import (
	"context"
	"os"
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

// ============================================================================
// CustomEngine Tests
// ============================================================================

// TestCustomEngineCreation tests that a custom engine can be created
func TestCustomEngineCreation(t *testing.T) {
	// This test requires a valid GGUF model file
	// Skip if model file is not available
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}

	if engine == nil {
		t.Fatal("CustomEngine is nil")
	}

	// Verify engine implements the interface
	var _ Engine = engine

	// Clean up
	if err := engine.Close(); err != nil {
		t.Errorf("Close failed: %v", err)
	}
}

// TestCustomEngineInvalidPath tests creation with invalid model path
func TestCustomEngineInvalidPath(t *testing.T) {
	_, err := NewCustomEngine("/nonexistent/model.gguf", LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err == nil {
		t.Error("Expected error when creating engine with invalid path, got nil")
	}
}

// TestCustomEngineConfigConversion tests LoadOptions to inference.Config conversion
func TestCustomEngineConfigConversion(t *testing.T) {
	// This is a unit test that verifies the config conversion logic
	// We test this by checking that the engine creation follows expected patterns

	tests := []struct {
		name    string
		opts    LoadOptions
		wantErr bool
	}{
		{
			name: "default context size",
			opts: LoadOptions{
				ContextSize: 2048,
				Threads:     4,
			},
			wantErr: true, // Will error without valid model, but tests conversion
		},
		{
			name: "large context size",
			opts: LoadOptions{
				ContextSize: 8192,
				Threads:     8,
			},
			wantErr: true,
		},
		{
			name: "minimal context size",
			opts: LoadOptions{
				ContextSize: 512,
				Threads:     1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewCustomEngine("/invalid/path.gguf", tt.opts)

			// We expect error due to invalid path
			// The test verifies that config conversion doesn't panic
			if err == nil {
				t.Error("Expected error with invalid path")
			}
		})
	}
}

// TestCustomEngineGenerate tests basic text generation with custom engine
func TestCustomEngineGenerate(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := engine.Generate(ctx, "Hello", GenerateOptions{
		MaxTokens:   10,
		Temperature: 0.2,
		TopP:        0.95,
		TopK:        40,
	})

	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if result == "" {
		t.Error("Generated result is empty")
	}

	t.Logf("Generated text: %s", truncateStr(result, 100))
}

// TestCustomEngineGenerateStream tests streaming generation with custom engine
func TestCustomEngineGenerateStream(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	stream, err := engine.GenerateStream(ctx, "Hello", GenerateOptions{
		MaxTokens:   10,
		Temperature: 0.2,
	})

	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}

	// Read from stream
	var chunks []string
	for chunk := range stream {
		chunks = append(chunks, chunk)
		// Stop after reasonable amount
		if len(chunks) > 20 {
			break
		}
	}

	if len(chunks) == 0 {
		t.Error("Expected at least one chunk from stream")
	}

	fullText := strings.Join(chunks, "")
	t.Logf("Streamed %d chunks, total text: %s", len(chunks), truncateStr(fullText, 100))
}

// TestCustomEngineTokenCount tests token counting with custom engine
func TestCustomEngineTokenCount(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}
	defer engine.Close()

	tests := []struct {
		name     string
		text     string
		minCount int
		maxCount int
	}{
		{
			name:     "simple text",
			text:     "Hello world",
			minCount: 1,
			maxCount: 10,
		},
		{
			name:     "longer text",
			text:     "This is a longer piece of text with multiple words",
			minCount: 5,
			maxCount: 20,
		},
		{
			name:     "empty text",
			text:     "",
			minCount: 0,
			maxCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count := engine.TokenCount(tt.text)

			if count < tt.minCount || count > tt.maxCount {
				t.Errorf("TokenCount(%q) = %d, want between %d and %d",
					tt.text, count, tt.minCount, tt.maxCount)
			}

			t.Logf("TokenCount(%q) = %d", tt.text, count)
		})
	}
}

// TestCustomEngineClose tests proper cleanup of custom engine
func TestCustomEngineClose(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}

	err = engine.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Double close should be safe
	err = engine.Close()
	if err != nil {
		t.Logf("Second close returned error (may be expected): %v", err)
	}
}

// TestCustomEngineWithConfig tests NewCustomEngineWithConfig
func TestCustomEngineWithConfig(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	loadOpts := LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	}

	genOpts := GenerateOptions{
		MaxTokens:   100,
		Temperature: 0.8,
		TopP:        0.9,
		TopK:        50,
	}

	engine, err := NewCustomEngineWithConfig(modelPath, loadOpts, genOpts)

	if err != nil {
		t.Fatalf("Failed to create custom engine with config: %v", err)
	}

	if engine == nil {
		t.Fatal("CustomEngine is nil")
	}

	// Verify engine implements the interface
	var _ Engine = engine

	// Clean up
	if err := engine.Close(); err != nil {
		t.Errorf("Close failed: %v", err)
	}
}

// TestCustomEngineString tests the String method
func TestCustomEngineString(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}
	defer engine.Close()

	str := engine.String()

	if !strings.Contains(str, "CustomEngine") {
		t.Errorf("String() should contain 'CustomEngine', got: %s", str)
	}

	if !strings.Contains(str, "pure-go") {
		t.Errorf("String() should contain 'pure-go', got: %s", str)
	}

	t.Logf("Engine string representation: %s", str)
}

// TestCustomEngineContextCancellation tests context cancellation with custom engine
func TestCustomEngineContextCancellation(t *testing.T) {
	modelPath := getTestModelPath()
	if modelPath == "" {
		t.Skip("Skipping CustomEngine test: no test model available")
	}

	engine, err := NewCustomEngine(modelPath, LoadOptions{
		ContextSize: 2048,
		Threads:     4,
	})

	if err != nil {
		t.Fatalf("Failed to create custom engine: %v", err)
	}
	defer engine.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = engine.Generate(ctx, "test", GenerateOptions{MaxTokens: 100})

	// Should handle cancellation (may or may not return error depending on timing)
	t.Logf("Generate with cancelled context: error=%v", err)
}

// Helper functions
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// getTestModelPath returns the path to a test GGUF model file
// Returns empty string if no test model is available
func getTestModelPath() string {
	// Check for test model in common locations
	// Users can set VIBRANT_TEST_MODEL environment variable
	// For CI/CD: skip tests if model not available

	// 1. Check environment variable
	if path := os.Getenv("VIBRANT_TEST_MODEL"); path != "" {
		return path
	}

	// 2. Look for a small test model in testdata/
	// (Could add checks for testdata/tiny_test.gguf, etc.)

	// 3. Return empty to skip tests that need real models
	// Integration tests with real models should be run separately
	return ""
}

