package transformer

import (
	"fmt"
	"time"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// Model represents a complete transformer language model
type Model struct {
	config *Config

	// Token embeddings
	embeddings *Embeddings

	// Transformer layers
	layers []*TransformerLayer

	// Output layer normalization
	outputNorm *RMSNorm

	// Output projection (language modeling head)
	outputWeight *tensor.Tensor // [hidden_dim, vocab_size]

	// Cached dequantized output weight (for quantized models)
	// This avoids repeated dequantization of the massive output matrix
	outputWeightCache *tensor.Tensor // [hidden_dim, vocab_size] in Float32

	// Target device for computation (CPU or GPU)
	device tensor.Device
}

// NewModel creates a complete model from GGUF file
func NewModel(ggufFile *gguf.GGUFFile) (*Model, error) {
	// Load configuration from GGUF metadata
	cfg, err := NewConfigFromGGUF(ggufFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	fmt.Printf("Loaded config: %s\n", cfg.String())

	// Create embeddings layer
	embeddings, err := NewEmbeddings(ggufFile, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	// Create all transformer layers
	layers := make([]*TransformerLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layer, err := NewTransformerLayer(ggufFile, cfg, i)
		if err != nil {
			return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	// Load output normalization (use eager dequantization for small tensors)
	outputNormWeight, err := loadTensorEager(ggufFile, "output_norm.weight")
	if err != nil {
		// Try alternative name
		outputNormWeight, err = loadTensorEager(ggufFile, "norm.weight")
		if err != nil {
			return nil, fmt.Errorf("failed to load output norm weight: %w", err)
		}
	}

	outputNorm, err := NewRMSNorm(outputNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create output norm: %w", err)
	}

	// Load output projection (LM head) - keep lazy for large matrix
	outputWeight, err := loadTensor(ggufFile, "output.weight")
	if err != nil {
		// Some models tie weights with embeddings
		outputWeight, err = loadTensor(ggufFile, "token_embd.weight")
		if err != nil {
			return nil, fmt.Errorf("failed to load output weight: %w", err)
		}
	}

	// Pre-transpose Float32 output weight for matmul optimization
	// This is done once at load time instead of repeatedly during generation
	// For quantized weights, transpose happens during dequantization (handled by cache)
	if outputWeight.DType() == tensor.Float32 {
		if err := outputWeight.PretransposeInPlace(); err != nil {
			return nil, fmt.Errorf("failed to pretranspose output weight: %w", err)
		}
	}

	return &Model{
		config:       cfg,
		embeddings:   embeddings,
		layers:       layers,
		outputNorm:   outputNorm,
		outputWeight: outputWeight,
	}, nil
}

// MoveToDevice moves all model weights to the specified device (CPU or GPU)
func (m *Model) MoveToDevice(device tensor.Device) error {
	// Store target device
	m.device = device

	// Move embeddings
	if err := m.embeddings.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move embeddings to device: %w", err)
	}

	// Move all layers
	for i, layer := range m.layers {
		if err := layer.MoveToDevice(device); err != nil {
			return fmt.Errorf("failed to move layer %d to device: %w", i, err)
		}
	}

	// Move output norm
	if err := m.outputNorm.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move output norm to device: %w", err)
	}

	// Move output weight
	if m.outputWeight != nil {
		gpuWeight, err := m.outputWeight.ToDevice(device)
		if err != nil {
			return fmt.Errorf("failed to move output weight to device: %w", err)
		}
		m.outputWeight.FreeGPU() // Free old GPU memory if any
		m.outputWeight = gpuWeight
	}

	return nil
}

// Forward performs a forward pass through the entire model
// Input: token IDs [batch_size, seq_len]
// Output: logits [batch_size, seq_len, vocab_size]
func (m *Model) Forward(tokenIDs [][]int, useCache bool) (*tensor.Tensor, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("empty token IDs")
	}

	seqLen := len(tokenIDs[0])

	// Create position indices
	// During decode with cache, positions must start from cache length
	positions := make([]int, seqLen)
	startPos := 0
	if useCache && len(m.layers) > 0 {
		startPos = m.layers[0].CacheLen()
	}
	for i := range positions {
		positions[i] = startPos + i
	}

	// 1. Embed tokens
	fmt.Printf("[MODEL] Starting embeddings...\n")
	hidden, err := m.embeddings.Forward(tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("embeddings failed: %w", err)
	}
	fmt.Printf("[MODEL] Embeddings complete, shape: %v\n", hidden.Shape())

	// Move embeddings to GPU if model is on GPU
	// (embeddings are computed on CPU and need to be transferred)
	if m.device == tensor.GPU {
		hidden, err = hidden.ToDevice(tensor.GPU)
		if err != nil {
			return nil, fmt.Errorf("failed to move embeddings to GPU: %w", err)
		}
	}

	// 2. Pass through all transformer layers
	fmt.Printf("[MODEL] Processing %d layers...\n", len(m.layers))
	for i, layer := range m.layers {
		layerStart := time.Now()
		fmt.Printf("[MODEL] Layer %d/%d starting...\n", i, len(m.layers))

		hidden, err = layer.Forward(hidden, positions, useCache)
		if err != nil {
			return nil, fmt.Errorf("layer %d failed: %w", i, err)
		}

		layerTime := time.Since(layerStart)
		fmt.Printf("[MODEL] Layer %d complete in %.3fs\n", i, layerTime.Seconds())
	}
	fmt.Printf("[MODEL] All layers complete\n")

	// 3. Final normalization
	fmt.Printf("[MODEL] Starting output norm...\n")
	hidden, err = m.outputNorm.Forward(hidden)
	if err != nil {
		return nil, fmt.Errorf("output norm failed: %w", err)
	}
	fmt.Printf("[MODEL] Output norm complete\n")

	// 4. Project to vocabulary (compute logits)
	// hidden: [batch, seq, hidden_dim], output_weight: [hidden_dim, vocab_size]
	// logits = hidden @ output_weight -> [batch, seq, vocab_size]
	fmt.Printf("[MODEL] Starting output projection...\n")
	shape := hidden.Shape()
	batchSize := shape[0]
	hiddenDim := shape[2]

	hiddenFlat := tensor.Reshape(hidden, []int{batchSize * seqLen, hiddenDim})
	fmt.Printf("[MODEL] Reshaped hidden: %v\n", hiddenFlat.Shape())
	fmt.Printf("[MODEL] Output weight shape: %v, dtype: %v\n", m.outputWeight.Shape(), m.outputWeight.DType())

	// Dequantize output weight once and cache it (HUGE performance win for large vocab)
	weightToUse := m.outputWeight
	if m.outputWeightCache == nil && m.outputWeight.DType() != tensor.Float32 {
		fmt.Printf("[MODEL] Output weight is quantized - dequantizing and caching (one-time cost)...\n")
		start := time.Now()

		// Dequantize the entire weight matrix
		m.outputWeightCache = tensor.Dequantize(m.outputWeight)

		elapsed := time.Since(start)
		fmt.Printf("[MODEL] Dequantization complete in %.2fs (%.1f MB cached)\n",
			elapsed.Seconds(), float64(m.outputWeightCache.Size()*4)/1e6)

		weightToUse = m.outputWeightCache
	} else if m.outputWeightCache != nil {
		// Use cached dequantized version
		fmt.Printf("[MODEL] Using cached dequantized weight (dtype: %v)\n", m.outputWeightCache.DType())
		weightToUse = m.outputWeightCache
	}

	fmt.Printf("[MODEL] MatMul input: hidden=%v, weight=%v (dtype: %v)\n",
		hiddenFlat.Shape(), weightToUse.Shape(), weightToUse.DType())
	fmt.Printf("[MODEL] Starting MatMul...\n")
	logitsFlat := tensor.MatMul(hiddenFlat, weightToUse)
	fmt.Printf("[MODEL] MatMul complete\n")
	logits := tensor.Reshape(logitsFlat, []int{batchSize, seqLen, m.config.VocabSize})
	fmt.Printf("[MODEL] Output projection complete, logits shape: %v\n", logits.Shape())

	return logits, nil
}

// Config returns the model configuration
func (m *Model) Config() *Config {
	return m.config
}

// NumLayers returns the number of transformer layers
func (m *Model) NumLayers() int {
	return len(m.layers)
}

// ClearCache clears the KV-cache for all layers
func (m *Model) ClearCache() {
	for _, layer := range m.layers {
		layer.ClearCache()
	}
}
