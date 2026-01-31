package transformer

import (
	"testing"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

func createMockConfig() *Config {
	return &Config{
		Architecture:    "qwen",
		ContextLength:   2048,
		VocabSize:       32000,
		HiddenDim:       512,
		NumLayers:       4,
		IntermediateDim: 2048,
		NumHeads:        8,
		NumKVHeads:      8,
		HeadDim:         64,
		RopeFreqBase:    10000.0,
		RopeScaling:     1.0,
		RMSNormEps:      1e-6,
	}
}

func TestConfig_Validate(t *testing.T) {
	cfg := createMockConfig()

	if err := cfg.Validate(); err != nil {
		t.Errorf("Valid config failed validation: %v", err)
	}
}

func TestConfig_ValidateInvalid(t *testing.T) {
	tests := []struct {
		name   string
		modify func(*Config)
	}{
		{"negative context", func(c *Config) { c.ContextLength = -1 }},
		{"zero vocab", func(c *Config) { c.VocabSize = 0 }},
		{"negative hidden", func(c *Config) { c.HiddenDim = -1 }},
		{"zero layers", func(c *Config) { c.NumLayers = 0 }},
		{"kv > heads", func(c *Config) { c.NumKVHeads = c.NumHeads + 1 }},
		{"dim not divisible", func(c *Config) { c.HiddenDim = 513; c.NumHeads = 8 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := createMockConfig()
			tt.modify(cfg)

			if err := cfg.Validate(); err == nil {
				t.Error("Expected validation error, got nil")
			}
		})
	}
}

func TestConfig_IsGQA(t *testing.T) {
	cfg := createMockConfig()

	// Standard MHA
	cfg.NumHeads = 8
	cfg.NumKVHeads = 8
	if cfg.IsGQA() {
		t.Error("Expected false for MHA (NumHeads == NumKVHeads)")
	}

	// Grouped-Query Attention
	cfg.NumKVHeads = 4
	if !cfg.IsGQA() {
		t.Error("Expected true for GQA (NumKVHeads < NumHeads)")
	}
}

func TestConfig_KVGroupSize(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 8
	cfg.NumKVHeads = 4

	if size := cfg.KVGroupSize(); size != 2 {
		t.Errorf("Expected KV group size 2, got %d", size)
	}
}

func TestConfig_String(t *testing.T) {
	cfg := createMockConfig()
	s := cfg.String()

	if s == "" {
		t.Error("Config string should not be empty")
	}

	// Should contain architecture
	if !contains(s, "qwen") {
		t.Errorf("Config string should contain architecture, got: %s", s)
	}
}

func TestRMSNorm_Forward(t *testing.T) {
	// Create a simple RMSNorm layer
	weight := tensor.NewTensor([]int{4}, tensor.Float32)
	for i := 0; i < 4; i++ {
		weight.Set(1.0, i)
	}

	norm, err := NewRMSNorm(weight, 1e-6)
	if err != nil {
		t.Fatalf("Failed to create RMSNorm: %v", err)
	}

	// Create input: [1, 2, 4] (batch=1, seq=2, hidden=4)
	input := tensor.NewTensor([]int{1, 2, 4}, tensor.Float32)
	for b := 0; b < 1; b++ {
		for s := 0; s < 2; s++ {
			for d := 0; d < 4; d++ {
				input.Set(float32(s*4+d+1), b, s, d)
			}
		}
	}

	output, err := norm.Forward(input)
	if err != nil {
		t.Errorf("RMSNorm forward failed: %v", err)
	}

	// Verify output shape matches input
	if !shapeEqual(output.Shape(), input.Shape()) {
		t.Errorf("Output shape %v != input shape %v", output.Shape(), input.Shape())
	}
}

func TestRMSNorm_InvalidShape(t *testing.T) {
	weight := tensor.NewTensor([]int{4}, tensor.Float32)
	norm, _ := NewRMSNorm(weight, 1e-6)

	// Invalid input: 2D instead of 3D
	input := tensor.NewTensor([]int{2, 4}, tensor.Float32)

	_, err := norm.Forward(input)
	if err == nil {
		t.Error("Expected error for invalid input shape")
	}
}

func TestRoPE_ApplyRotation(t *testing.T) {
	rope := NewRoPE(64, 10000.0, 2048)

	// Create Q tensor: [1, 4, 3, 64] (batch=1, heads=4, seq=3, dim=64)
	q := tensor.NewTensor([]int{1, 4, 3, 64}, tensor.Float32)

	positions := []int{0, 1, 2}

	output, err := rope.ApplyRotation(q, positions)
	if err != nil {
		t.Errorf("RoPE failed: %v", err)
	}

	// Verify output shape matches input
	if !shapeEqual(output.Shape(), q.Shape()) {
		t.Errorf("Output shape %v != input shape %v", output.Shape(), q.Shape())
	}
}

func TestNewConfigFromGGUF_NoArch(t *testing.T) {
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
	}

	// No architecture key
	_, err := NewConfigFromGGUF(ggufFile)
	if err == nil {
		t.Error("Expected error for missing architecture")
	}
}

func TestNewConfigFromGGUF_Complete(t *testing.T) {
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
	}

	// Set minimal required metadata
	ggufFile.Metadata["general.architecture"] = "qwen"
	ggufFile.Metadata["qwen.context_length"] = uint32(2048)
	ggufFile.Metadata["qwen.embedding_length"] = uint32(512)
	ggufFile.Metadata["qwen.block_count"] = uint32(4)
	ggufFile.Metadata["qwen.attention.head_count"] = uint32(8)
	ggufFile.Metadata["qwen.attention.head_count_kv"] = uint32(4)
	ggufFile.Metadata["qwen.feed_forward_length"] = uint32(2048)
	ggufFile.Metadata["qwen.rope.freq_base"] = float32(10000.0)
	ggufFile.Metadata["qwen.attention.layer_norm_rms_epsilon"] = float64(1e-6)
	ggufFile.Metadata["tokenizer.ggml.tokens"] = []interface{}{"a", "b", "c"}

	cfg, err := NewConfigFromGGUF(ggufFile)
	if err != nil {
		t.Fatalf("NewConfigFromGGUF failed: %v", err)
	}

	if cfg.Architecture != "qwen" {
		t.Errorf("Expected architecture 'qwen', got '%s'", cfg.Architecture)
	}

	if cfg.ContextLength != 2048 {
		t.Errorf("Expected context length 2048, got %d", cfg.ContextLength)
	}

	if cfg.VocabSize != 3 {
		t.Errorf("Expected vocab size 3, got %d", cfg.VocabSize)
	}

	if cfg.NumHeads != 8 {
		t.Errorf("Expected 8 heads, got %d", cfg.NumHeads)
	}

	if cfg.NumKVHeads != 4 {
		t.Errorf("Expected 4 KV heads, got %d", cfg.NumKVHeads)
	}
}

func TestEmbeddings_InvalidTokenID(t *testing.T) {
	cfg := createMockConfig()

	// Create mock weight tensor
	weight := tensor.NewTensor([]int{cfg.VocabSize, cfg.HiddenDim}, tensor.Float32)

	emb := &Embeddings{
		weight:    weight,
		vocabSize: cfg.VocabSize,
		hiddenDim: cfg.HiddenDim,
	}

	// Test with out-of-range token ID
	tokenIDs := [][]int{{0, 999999}} // 999999 is out of range

	_, err := emb.Forward(tokenIDs)
	if err == nil {
		t.Error("Expected error for out-of-range token ID")
	}
}

func TestEmbeddings_EmptyInput(t *testing.T) {
	cfg := createMockConfig()
	weight := tensor.NewTensor([]int{cfg.VocabSize, cfg.HiddenDim}, tensor.Float32)

	emb := &Embeddings{
		weight:    weight,
		vocabSize: cfg.VocabSize,
		hiddenDim: cfg.HiddenDim,
	}

	_, err := emb.Forward([][]int{})
	if err == nil {
		t.Error("Expected error for empty input")
	}
}

// Helper functions
func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Test Attention layer forward pass
func TestAttention_ForwardShape(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 4
	cfg.NumKVHeads = 4
	cfg.HeadDim = 32
	cfg.HiddenDim = 128

	// Create mock attention layer
	attn := &Attention{
		wq:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumHeads * cfg.HeadDim}, tensor.Float32),
		wk:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wv:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wo:         tensor.NewTensor([]int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenDim}, tensor.Float32),
		numHeads:   cfg.NumHeads,
		numKVHeads: cfg.NumKVHeads,
		headDim:    cfg.HeadDim,
		hiddenDim:  cfg.HiddenDim,
		rope:       NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength),
	}

	// Initialize weights with small values
	initTensor(attn.wq, 0.01)
	initTensor(attn.wk, 0.01)
	initTensor(attn.wv, 0.01)
	initTensor(attn.wo, 0.01)

	// Create input tensor [batch=1, seq=4, hidden=128]
	batchSize := 1
	seqLen := 4
	input := tensor.NewTensor([]int{batchSize, seqLen, cfg.HiddenDim}, tensor.Float32)
	initTensor(input, 0.1)

	positions := []int{0, 1, 2, 3}

	// Forward pass
	output, err := attn.Forward(input, positions, false)
	if err != nil {
		t.Fatalf("Attention forward failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{batchSize, seqLen, cfg.HiddenDim}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}
}

// Test Attention layer with GQA
func TestAttention_ForwardGQA(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 8
	cfg.NumKVHeads = 2 // GQA: 4 query heads per KV head
	cfg.HeadDim = 32
	cfg.HiddenDim = 256

	// Create mock attention layer with GQA
	attn := &Attention{
		wq:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumHeads * cfg.HeadDim}, tensor.Float32),
		wk:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wv:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wo:         tensor.NewTensor([]int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenDim}, tensor.Float32),
		numHeads:   cfg.NumHeads,
		numKVHeads: cfg.NumKVHeads,
		headDim:    cfg.HeadDim,
		hiddenDim:  cfg.HiddenDim,
		rope:       NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength),
	}

	initTensor(attn.wq, 0.01)
	initTensor(attn.wk, 0.01)
	initTensor(attn.wv, 0.01)
	initTensor(attn.wo, 0.01)

	// Create input tensor [batch=2, seq=3, hidden=256]
	batchSize := 2
	seqLen := 3
	input := tensor.NewTensor([]int{batchSize, seqLen, cfg.HiddenDim}, tensor.Float32)
	initTensor(input, 0.1)

	positions := []int{0, 1, 2}

	// Forward pass with GQA
	output, err := attn.Forward(input, positions, false)
	if err != nil {
		t.Fatalf("Attention GQA forward failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{batchSize, seqLen, cfg.HiddenDim}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}
}

// Test FeedForward layer forward pass
func TestFeedForward_Forward(t *testing.T) {
	cfg := createMockConfig()
	cfg.HiddenDim = 128
	cfg.IntermediateDim = 512

	// Create mock FFN layer
	ffn := &FeedForward{
		gate:            tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		up:              tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		down:            tensor.NewTensor([]int{cfg.IntermediateDim, cfg.HiddenDim}, tensor.Float32),
		hiddenDim:       cfg.HiddenDim,
		intermediateDim: cfg.IntermediateDim,
	}

	initTensor(ffn.gate, 0.01)
	initTensor(ffn.up, 0.01)
	initTensor(ffn.down, 0.01)

	// Create input tensor [batch=1, seq=4, hidden=128]
	batchSize := 1
	seqLen := 4
	input := tensor.NewTensor([]int{batchSize, seqLen, cfg.HiddenDim}, tensor.Float32)
	initTensor(input, 0.1)

	// Forward pass
	output, err := ffn.Forward(input)
	if err != nil {
		t.Fatalf("FeedForward forward failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{batchSize, seqLen, cfg.HiddenDim}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}
}

// Test FeedForward with invalid input shape
func TestFeedForward_InvalidShape(t *testing.T) {
	cfg := createMockConfig()
	ffn := &FeedForward{
		gate:            tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		up:              tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		down:            tensor.NewTensor([]int{cfg.IntermediateDim, cfg.HiddenDim}, tensor.Float32),
		hiddenDim:       cfg.HiddenDim,
		intermediateDim: cfg.IntermediateDim,
	}

	// Create 2D input (invalid - expects 3D)
	input := tensor.NewTensor([]int{4, cfg.HiddenDim}, tensor.Float32)

	_, err := ffn.Forward(input)
	if err == nil {
		t.Error("Expected error for invalid input shape, got nil")
	}
}

// Test TransformerLayer forward pass
func TestTransformerLayer_Forward(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 4
	cfg.NumKVHeads = 4
	cfg.HeadDim = 32
	cfg.HiddenDim = 128
	cfg.IntermediateDim = 512

	// Create mock layer components
	attn := &Attention{
		wq:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumHeads * cfg.HeadDim}, tensor.Float32),
		wk:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wv:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wo:         tensor.NewTensor([]int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenDim}, tensor.Float32),
		numHeads:   cfg.NumHeads,
		numKVHeads: cfg.NumKVHeads,
		headDim:    cfg.HeadDim,
		hiddenDim:  cfg.HiddenDim,
		rope:       NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength),
	}

	initTensor(attn.wq, 0.01)
	initTensor(attn.wk, 0.01)
	initTensor(attn.wv, 0.01)
	initTensor(attn.wo, 0.01)

	ffn := &FeedForward{
		gate:            tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		up:              tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
		down:            tensor.NewTensor([]int{cfg.IntermediateDim, cfg.HiddenDim}, tensor.Float32),
		hiddenDim:       cfg.HiddenDim,
		intermediateDim: cfg.IntermediateDim,
	}

	initTensor(ffn.gate, 0.01)
	initTensor(ffn.up, 0.01)
	initTensor(ffn.down, 0.01)

	attnNormWeight := tensor.NewTensor([]int{cfg.HiddenDim}, tensor.Float32)
	ffnNormWeight := tensor.NewTensor([]int{cfg.HiddenDim}, tensor.Float32)
	initTensor(attnNormWeight, 1.0)
	initTensor(ffnNormWeight, 1.0)

	attnNorm, _ := NewRMSNorm(attnNormWeight, cfg.RMSNormEps)
	ffnNorm, _ := NewRMSNorm(ffnNormWeight, cfg.RMSNormEps)

	layer := &TransformerLayer{
		layerIdx: 0,
		attn:     attn,
		ffn:      ffn,
		attnNorm: attnNorm,
		ffnNorm:  ffnNorm,
	}

	// Create input tensor [batch=1, seq=4, hidden=128]
	batchSize := 1
	seqLen := 4
	input := tensor.NewTensor([]int{batchSize, seqLen, cfg.HiddenDim}, tensor.Float32)
	initTensor(input, 0.1)

	positions := []int{0, 1, 2, 3}

	// Forward pass
	output, err := layer.Forward(input, positions, false)
	if err != nil {
		t.Fatalf("TransformerLayer forward failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{batchSize, seqLen, cfg.HiddenDim}
	if !shapesEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape())
	}
}

// Helper: initialize tensor with a constant value
func initTensor(t *tensor.Tensor, val float64) {
	shape := t.Shape()
	switch len(shape) {
	case 1:
		for i := 0; i < shape[0]; i++ {
			t.Set(float32(val), i)
		}
	case 2:
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				t.Set(float32(val), i, j)
			}
		}
	case 3:
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				for k := 0; k < shape[2]; k++ {
					t.Set(float32(val), i, j, k)
				}
			}
		}
	case 4:
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				for k := 0; k < shape[2]; k++ {
					for l := 0; l < shape[3]; l++ {
						t.Set(float32(val), i, j, k, l)
					}
				}
			}
		}
	}
}

// Helper: compare shapes
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Test cache operations
func TestAttention_CacheOperations(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 4
	cfg.NumKVHeads = 4
	cfg.HeadDim = 32
	cfg.HiddenDim = 128

	// Create mock attention layer
	attn := &Attention{
		wq:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumHeads * cfg.HeadDim}, tensor.Float32),
		wk:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wv:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
		wo:         tensor.NewTensor([]int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenDim}, tensor.Float32),
		numHeads:   cfg.NumHeads,
		numKVHeads: cfg.NumKVHeads,
		headDim:    cfg.HeadDim,
		hiddenDim:  cfg.HiddenDim,
		rope:       NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength),
	}

	initTensor(attn.wq, 0.01)
	initTensor(attn.wk, 0.01)
	initTensor(attn.wv, 0.01)
	initTensor(attn.wo, 0.01)

	// Test 1: First forward pass builds cache
	input1 := tensor.NewTensor([]int{1, 3, cfg.HiddenDim}, tensor.Float32)
	initTensor(input1, 0.1)
	positions1 := []int{0, 1, 2}

	_, err := attn.Forward(input1, positions1, true) // useCache=true
	if err != nil {
		t.Fatalf("First forward pass failed: %v", err)
	}

	// Check cache was created
	if attn.kCache == nil || attn.vCache == nil {
		t.Error("Cache should be initialized after first forward pass")
	}

	if attn.cacheLen != 3 {
		t.Errorf("Expected cache length 3, got %d", attn.cacheLen)
	}

	// Test 2: Second forward pass appends to cache
	input2 := tensor.NewTensor([]int{1, 2, cfg.HiddenDim}, tensor.Float32)
	initTensor(input2, 0.2)
	positions2 := []int{3, 4}

	_, err = attn.Forward(input2, positions2, true) // useCache=true
	if err != nil {
		t.Fatalf("Second forward pass failed: %v", err)
	}

	// Check cache grew
	if attn.cacheLen != 5 {
		t.Errorf("Expected cache length 5 after append, got %d", attn.cacheLen)
	}

	// Test 3: Clear cache
	attn.ClearCache()

	if attn.kCache != nil || attn.vCache != nil {
		t.Error("Cache should be nil after clearing")
	}

	if attn.cacheLen != 0 {
		t.Errorf("Expected cache length 0 after clear, got %d", attn.cacheLen)
	}
}

func TestModel_ClearCache(t *testing.T) {
	cfg := createMockConfig()
	cfg.NumHeads = 4
	cfg.NumKVHeads = 4
	cfg.HeadDim = 32
	cfg.HiddenDim = 128
	cfg.NumLayers = 2

	// Create mock model with 2 layers
	layers := make([]*TransformerLayer, 2)
	for i := 0; i < 2; i++ {
		attn := &Attention{
			wq:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumHeads * cfg.HeadDim}, tensor.Float32),
			wk:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
			wv:         tensor.NewTensor([]int{cfg.HiddenDim, cfg.NumKVHeads * cfg.HeadDim}, tensor.Float32),
			wo:         tensor.NewTensor([]int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenDim}, tensor.Float32),
			numHeads:   cfg.NumHeads,
			numKVHeads: cfg.NumKVHeads,
			headDim:    cfg.HeadDim,
			hiddenDim:  cfg.HiddenDim,
			rope:       NewRoPE(cfg.HeadDim, cfg.RopeFreqBase, cfg.ContextLength),
			cacheLen:   5, // Simulate cached tokens
			kCache:     tensor.NewTensor([]int{1, 4, 5, 32}, tensor.Float32),
			vCache:     tensor.NewTensor([]int{1, 4, 5, 32}, tensor.Float32),
		}

		ffn := &FeedForward{
			gate:            tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
			up:              tensor.NewTensor([]int{cfg.HiddenDim, cfg.IntermediateDim}, tensor.Float32),
			down:            tensor.NewTensor([]int{cfg.IntermediateDim, cfg.HiddenDim}, tensor.Float32),
			hiddenDim:       cfg.HiddenDim,
			intermediateDim: cfg.IntermediateDim,
		}

		attnNormWeight := tensor.NewTensor([]int{cfg.HiddenDim}, tensor.Float32)
		ffnNormWeight := tensor.NewTensor([]int{cfg.HiddenDim}, tensor.Float32)
		initTensor(attnNormWeight, 1.0)
		initTensor(ffnNormWeight, 1.0)

		attnNorm, _ := NewRMSNorm(attnNormWeight, cfg.RMSNormEps)
		ffnNorm, _ := NewRMSNorm(ffnNormWeight, cfg.RMSNormEps)

		layers[i] = &TransformerLayer{
			layerIdx: i,
			attn:     attn,
			ffn:      ffn,
			attnNorm: attnNorm,
			ffnNorm:  ffnNorm,
		}
	}

	model := &Model{
		config: cfg,
		layers: layers,
	}

	// Clear all caches
	model.ClearCache()

	// Verify all caches are cleared
	for i, layer := range model.layers {
		if layer.attn.kCache != nil || layer.attn.vCache != nil {
			t.Errorf("Layer %d cache should be nil after model.ClearCache()", i)
		}
		if layer.attn.cacheLen != 0 {
			t.Errorf("Layer %d cache length should be 0, got %d", i, layer.attn.cacheLen)
		}
	}
}

func TestConcatenateSeqDim(t *testing.T) {
	// Cached: [1, 2, 3, 4] (batch=1, heads=2, seq=3, dim=4)
	cached := tensor.NewTensor([]int{1, 2, 3, 4}, tensor.Float32)
	for b := 0; b < 1; b++ {
		for h := 0; h < 2; h++ {
			for s := 0; s < 3; s++ {
				for d := 0; d < 4; d++ {
					cached.Set(float32(s*10+d), b, h, s, d)
				}
			}
		}
	}

	// New: [1, 2, 2, 4] (batch=1, heads=2, seq=2, dim=4)
	new := tensor.NewTensor([]int{1, 2, 2, 4}, tensor.Float32)
	for b := 0; b < 1; b++ {
		for h := 0; h < 2; h++ {
			for s := 0; s < 2; s++ {
				for d := 0; d < 4; d++ {
					new.Set(float32(100+s*10+d), b, h, s, d)
				}
			}
		}
	}

	// Concatenate
	result := concatenateSeqDim(cached, new, 3)

	// Check shape: [1, 2, 5, 4]
	shape := result.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 5 || shape[3] != 4 {
		t.Errorf("Expected shape [1,2,5,4], got %v", shape)
	}

	// Check values: first 3 positions should be from cached
	for s := 0; s < 3; s++ {
		for d := 0; d < 4; d++ {
			expected := float32(s*10 + d)
			actual := result.At(0, 0, s, d)
			if actual != expected {
				t.Errorf("Cached values at [0,0,%d,%d]: expected %f, got %f", s, d, expected, actual)
			}
		}
	}

	// Check values: last 2 positions should be from new
	for s := 0; s < 2; s++ {
		for d := 0; d < 4; d++ {
			expected := float32(100 + s*10 + d)
			actual := result.At(0, 0, 3+s, d)
			if actual != expected {
				t.Errorf("New values at [0,0,%d,%d]: expected %f, got %f", 3+s, d, expected, actual)
			}
		}
	}
}
