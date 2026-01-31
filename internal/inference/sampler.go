package inference

import (
	"math"
	"math/rand"
	"sort"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// Sampler implements various sampling strategies for token generation
type Sampler struct {
	temperature float32     // Temperature for sampling (0 = greedy, >1 = more random)
	topP        float32     // Top-P (nucleus) sampling threshold
	topK        int         // Top-K sampling (keep only top K tokens)
	rng         *rand.Rand  // Random number generator
}

// NewSampler creates a new sampler with specified parameters
func NewSampler(temperature float32, topP float32, topK int, seed int64) *Sampler {
	return &Sampler{
		temperature: temperature,
		topP:        topP,
		topK:        topK,
		rng:         rand.New(rand.NewSource(seed)),
	}
}

// Sample selects the next token from logits using configured sampling strategy
// logits: [vocab_size] tensor of unnormalized log probabilities
// Returns: token ID
func (s *Sampler) Sample(logits *tensor.Tensor) int {
	// Greedy sampling (deterministic)
	if s.temperature == 0 {
		return s.SampleGreedy(logits)
	}

	// Clone logits to avoid modifying original
	logitsCopy := cloneTensor1D(logits)

	// 1. Apply temperature scaling
	s.applyTemperature(logitsCopy)

	// 2. Apply top-K filtering
	if s.topK > 0 {
		s.applyTopK(logitsCopy)
	}

	// 3. Apply top-P (nucleus) filtering
	if s.topP < 1.0 {
		s.applyTopP(logitsCopy)
	}

	// 4. Sample from filtered distribution
	return s.sampleMultinomial(logitsCopy)
}

// SampleGreedy returns the token with highest logit value (deterministic)
func (s *Sampler) SampleGreedy(logits *tensor.Tensor) int {
	vocabSize := logits.Size()
	maxIdx := 0
	maxVal := logits.At(0)

	for i := 1; i < vocabSize; i++ {
		val := logits.At(i)
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}

// applyTemperature scales logits by temperature
// Lower temperature -> peakier distribution (more confident)
// Higher temperature -> flatter distribution (more random)
func (s *Sampler) applyTemperature(logits *tensor.Tensor) {
	if s.temperature == 1.0 {
		return // No scaling needed
	}

	vocabSize := logits.Size()
	for i := 0; i < vocabSize; i++ {
		val := logits.At(i)
		logits.Set(float32(float64(val)/float64(s.temperature)), i)
	}
}

// applyTopK keeps only the top K tokens, setting others to -inf
func (s *Sampler) applyTopK(logits *tensor.Tensor) {
	vocabSize := logits.Size()

	if s.topK <= 0 || s.topK >= vocabSize {
		return // No filtering
	}

	// Create sorted indices
	indices := make([]int, vocabSize)
	values := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		indices[i] = i
		values[i] = logits.At(i)
	}

	// Sort by value descending
	sort.Slice(indices, func(i, j int) bool {
		return values[indices[i]] > values[indices[j]]
	})

	// Get k-th largest value
	threshold := values[indices[s.topK-1]]

	// Mask out values below threshold
	for i := 0; i < vocabSize; i++ {
		if logits.At(i) < threshold {
			logits.Set(float32(math.Inf(-1)), i)
		}
	}
}

// applyTopP keeps tokens with cumulative probability mass <= P (nucleus sampling)
func (s *Sampler) applyTopP(logits *tensor.Tensor) {
	if s.topP >= 1.0 {
		return // No filtering
	}

	vocabSize := logits.Size()

	// Convert logits to probabilities using softmax
	probs := s.softmax(logits)

	// Sort probabilities in descending order
	indices := make([]int, vocabSize)
	values := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		indices[i] = i
		values[i] = probs[i]
	}

	sort.Slice(indices, func(i, j int) bool {
		return values[indices[i]] > values[indices[j]]
	})

	// Compute cumulative sum and find cutoff
	cumsum := float32(0)
	cutoffIdx := vocabSize // Default: keep all

	for i, idx := range indices {
		cumsum += values[idx]
		if cumsum >= s.topP {
			cutoffIdx = i + 1
			break
		}
	}

	// Mask out tokens after cutoff
	keep := make(map[int]bool)
	for i := 0; i < cutoffIdx; i++ {
		keep[indices[i]] = true
	}

	for i := 0; i < vocabSize; i++ {
		if !keep[i] {
			logits.Set(float32(math.Inf(-1)), i)
		}
	}
}

// sampleMultinomial samples from a categorical distribution defined by logits
func (s *Sampler) sampleMultinomial(logits *tensor.Tensor) int {
	vocabSize := logits.Size()

	// Convert logits to probabilities
	probs := s.softmax(logits)

	// Sample using cumulative distribution
	r := s.rng.Float32()
	cumsum := float32(0)

	for i := 0; i < vocabSize; i++ {
		cumsum += probs[i]
		if r <= cumsum {
			return i
		}
	}

	// Fallback (shouldn't happen with proper normalization)
	return vocabSize - 1
}

// softmax converts logits to probabilities with numerical stability
func (s *Sampler) softmax(logits *tensor.Tensor) []float32 {
	vocabSize := logits.Size()
	probs := make([]float32, vocabSize)

	// Find max for numerical stability
	maxVal := float32(math.Inf(-1))
	for i := 0; i < vocabSize; i++ {
		val := logits.At(i)
		if !math.IsInf(float64(val), -1) && val > maxVal {
			maxVal = val
		}
	}

	// Compute exp(x - max) and sum
	sum := float32(0)
	for i := 0; i < vocabSize; i++ {
		val := logits.At(i)
		if math.IsInf(float64(val), -1) {
			probs[i] = 0
		} else {
			probs[i] = float32(math.Exp(float64(val - maxVal)))
		}
		sum += probs[i]
	}

	// Normalize
	if sum > 0 {
		for i := 0; i < vocabSize; i++ {
			probs[i] /= sum
		}
	} else {
		// Uniform distribution as fallback
		for i := 0; i < vocabSize; i++ {
			probs[i] = 1.0 / float32(vocabSize)
		}
	}

	return probs
}

// Helper: clone a 1D tensor
func cloneTensor1D(t *tensor.Tensor) *tensor.Tensor {
	size := t.Size()
	result := tensor.NewTensor([]int{size}, t.DType())

	for i := 0; i < size; i++ {
		result.Set(t.At(i), i)
	}

	return result
}
