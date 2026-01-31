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
