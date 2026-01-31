package inference

import (
	"math"
	"testing"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// ============================================================================
// Sampler Tests
// ============================================================================

func TestSampler_Greedy(t *testing.T) {
	sampler := NewSampler(0, 1.0, 0, 42) // temperature=0 -> greedy

	logits := tensor.NewTensor([]int{5}, tensor.Float32)
	logits.Set(1.0, 0)
	logits.Set(3.5, 1) // max
	logits.Set(2.0, 2)
	logits.Set(1.5, 3)
	logits.Set(0.5, 4)

	token := sampler.Sample(logits)
	if token != 1 {
		t.Errorf("Greedy sampling should select max logit, expected 1, got %d", token)
	}
}

func TestSampler_GreedyDirect(t *testing.T) {
	sampler := NewSampler(1.0, 1.0, 0, 42)

	logits := tensor.NewTensor([]int{4}, tensor.Float32)
	logits.Set(2.0, 0)
	logits.Set(5.0, 1) // max
	logits.Set(3.0, 2)
	logits.Set(1.0, 3)

	token := sampler.SampleGreedy(logits)
	if token != 1 {
		t.Errorf("Expected token 1, got %d", token)
	}
}

func TestSampler_Temperature(t *testing.T) {
	// Test that temperature affects distribution
	sampler := NewSampler(0.5, 1.0, 0, 42) // low temp -> more peaked

	logits := tensor.NewTensor([]int{3}, tensor.Float32)
	logits.Set(1.0, 0)
	logits.Set(2.0, 1)
	logits.Set(1.5, 2)

	// With low temperature, should favor higher logits more strongly
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		logitsCopy := cloneTensor1D(logits)
		token := sampler.Sample(logitsCopy)
		counts[token]++
	}

	// Token 1 (highest logit) should be sampled most often
	if counts[1] < 50 {
		t.Errorf("With low temperature, highest logit should be sampled most often, got counts: %v", counts)
	}
}

func TestSampler_TopK(t *testing.T) {
	sampler := NewSampler(1.0, 1.0, 2, 42) // top-k=2

	logits := tensor.NewTensor([]int{5}, tensor.Float32)
	logits.Set(1.0, 0)
	logits.Set(4.0, 1) // top-1
	logits.Set(3.0, 2) // top-2
	logits.Set(2.0, 3) // filtered out
	logits.Set(0.5, 4) // filtered out

	// Sample many times, should only get tokens 1 or 2
	for i := 0; i < 50; i++ {
		logitsCopy := cloneTensor1D(logits)
		token := sampler.Sample(logitsCopy)
		if token != 1 && token != 2 {
			t.Errorf("Top-K=2 should only sample tokens 1 or 2, got %d", token)
		}
	}
}

func TestSampler_TopP(t *testing.T) {
	sampler := NewSampler(1.0, 0.5, 0, 42) // top-p=0.5 (nucleus)

	logits := tensor.NewTensor([]int{4}, tensor.Float32)
	logits.Set(10.0, 0) // Very high probability after softmax
	logits.Set(1.0, 1)
	logits.Set(0.5, 2)
	logits.Set(0.1, 3)

	// With top-p=0.5, should heavily favor token 0
	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		logitsCopy := cloneTensor1D(logits)
		token := sampler.Sample(logitsCopy)
		counts[token]++
	}

	// Token 0 should dominate due to high logit
	if counts[0] < 80 {
		t.Errorf("Top-P should favor high probability tokens, got counts: %v", counts)
	}
}

func TestSampler_Softmax(t *testing.T) {
	sampler := NewSampler(1.0, 1.0, 0, 42)

	logits := tensor.NewTensor([]int{3}, tensor.Float32)
	logits.Set(1.0, 0)
	logits.Set(2.0, 1)
	logits.Set(3.0, 2)

	probs := sampler.softmax(logits)

	// Check probabilities sum to 1
	sum := float32(0)
	for _, p := range probs {
		sum += p
	}

	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("Softmax probabilities should sum to 1, got %f", sum)
	}

	// Check probabilities are in ascending order (since logits are ascending)
	if !(probs[0] < probs[1] && probs[1] < probs[2]) {
		t.Errorf("Softmax should preserve order, got probs: %v", probs)
	}
}

func TestSampler_SoftmaxWithInf(t *testing.T) {
	sampler := NewSampler(1.0, 1.0, 0, 42)

	logits := tensor.NewTensor([]int{4}, tensor.Float32)
	logits.Set(1.0, 0)
	logits.Set(float32(math.Inf(-1)), 1) // -inf (masked)
	logits.Set(2.0, 2)
	logits.Set(float32(math.Inf(-1)), 3) // -inf (masked)

	probs := sampler.softmax(logits)

	// Masked positions should have 0 probability
	if probs[1] != 0 || probs[3] != 0 {
		t.Errorf("Masked logits should have 0 probability, got: %v", probs)
	}

	// Non-masked probabilities should sum to 1
	sum := probs[0] + probs[2]
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("Non-masked probabilities should sum to 1, got %f", sum)
	}
}

// ============================================================================
// Cache Tests
// ============================================================================

// Note: concatenateSeqDim is tested in transformer package
// where it's defined. Skipping duplicate test here.

func TestCloneTensor1D(t *testing.T) {
	original := tensor.NewTensor([]int{5}, tensor.Float32)
	for i := 0; i < 5; i++ {
		original.Set(float32(i*2), i)
	}

	clone := cloneTensor1D(original)

	// Check values are the same
	for i := 0; i < 5; i++ {
		if clone.At(i) != original.At(i) {
			t.Errorf("Clone mismatch at index %d", i)
		}
	}

	// Modify clone shouldn't affect original
	clone.Set(999.0, 0)
	if original.At(0) == 999.0 {
		t.Error("Modifying clone affected original")
	}
}

// ============================================================================
// Engine Tests (Integration)
// ============================================================================

func TestEngine_TokenCount(t *testing.T) {
	// Note: This test would require a real GGUF file with tokenizer
	// For now, we'll skip it or create a mock
	t.Skip("Requires real GGUF file with tokenizer")
}

func TestEngine_Close(t *testing.T) {
	// Note: This test would require a real GGUF file
	// For now, we'll skip it
	t.Skip("Requires real GGUF file")
}

// ============================================================================
// Helper Function Tests
// ============================================================================

func TestIsStopToken(t *testing.T) {
	engine := &Engine{}

	stopTokens := []int{1, 5, 10}

	if !engine.isStopToken(1, stopTokens) {
		t.Error("Expected token 1 to be a stop token")
	}

	if !engine.isStopToken(10, stopTokens) {
		t.Error("Expected token 10 to be a stop token")
	}

	if engine.isStopToken(3, stopTokens) {
		t.Error("Expected token 3 to NOT be a stop token")
	}

	if engine.isStopToken(100, stopTokens) {
		t.Error("Expected token 100 to NOT be a stop token")
	}
}

func TestExtractLastTokenLogits(t *testing.T) {
	// Create logits: [1, 3, 5] (batch=1, seq=3, vocab=5)
	logits := tensor.NewTensor([]int{1, 3, 5}, tensor.Float32)

	// Fill with test values
	for s := 0; s < 3; s++ {
		for v := 0; v < 5; v++ {
			logits.Set(float32(s*10+v), 0, s, v)
		}
	}

	// Extract last token (s=2)
	shape := logits.Shape()
	lastTokenLogits := extractLastTokenLogits(logits, shape)

	// Check shape
	lastShape := lastTokenLogits.Shape()
	if len(lastShape) != 1 || lastShape[0] != 5 {
		t.Errorf("Expected shape [5], got %v", lastShape)
	}

	// Check values (should be s=2 values: 20, 21, 22, 23, 24)
	for v := 0; v < 5; v++ {
		expected := float32(20 + v)
		actual := lastTokenLogits.At(v)
		if actual != expected {
			t.Errorf("At vocab %d: expected %f, got %f", v, expected, actual)
		}
	}
}


// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkSampler_Greedy(b *testing.B) {
	sampler := NewSampler(0, 1.0, 0, 42)
	logits := tensor.NewTensor([]int{32000}, tensor.Float32) // Typical vocab size

	for i := 0; i < 32000; i++ {
		logits.Set(float32(i%100), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.SampleGreedy(logits)
	}
}

func BenchmarkSampler_Temperature(b *testing.B) {
	sampler := NewSampler(0.7, 1.0, 0, 42)
	logits := tensor.NewTensor([]int{32000}, tensor.Float32)

	for i := 0; i < 32000; i++ {
		logits.Set(float32(i%100), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logitsCopy := cloneTensor1D(logits)
		_ = sampler.Sample(logitsCopy)
	}
}

func BenchmarkSampler_TopK(b *testing.B) {
	sampler := NewSampler(0.7, 1.0, 50, 42)
	logits := tensor.NewTensor([]int{32000}, tensor.Float32)

	for i := 0; i < 32000; i++ {
		logits.Set(float32(i%100), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logitsCopy := cloneTensor1D(logits)
		_ = sampler.Sample(logitsCopy)
	}
}

func BenchmarkSampler_TopP(b *testing.B) {
	sampler := NewSampler(0.7, 0.9, 0, 42)
	logits := tensor.NewTensor([]int{32000}, tensor.Float32)

	for i := 0; i < 32000; i++ {
		logits.Set(float32(i%100), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		logitsCopy := cloneTensor1D(logits)
		_ = sampler.Sample(logitsCopy)
	}
}

// Note: concatenateSeqDim benchmark is in transformer package

func BenchmarkSoftmax(b *testing.B) {
	sampler := NewSampler(1.0, 1.0, 0, 42)
	logits := tensor.NewTensor([]int{32000}, tensor.Float32)

	for i := 0; i < 32000; i++ {
		logits.Set(float32(i%100), i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.softmax(logits)
	}
}
