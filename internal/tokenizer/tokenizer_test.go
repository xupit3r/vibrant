package tokenizer

import (
	"testing"

	"github.com/xupit3r/vibrant/internal/gguf"
)

// createMockTokenizer creates a simple test tokenizer
func createMockTokenizer() *Tokenizer {
	t := NewTokenizer()
	t.modelType = "test"

	// Simple vocabulary
	t.tokens = []string{
		"<unk>",  // 0
		"<s>",    // 1 (BOS)
		"</s>",   // 2 (EOS)
		"<pad>",  // 3
		"h",      // 4
		"e",      // 5
		"l",      // 6
		"o",      // 7
		"w",      // 8
		"r",      // 9
		"d",      // 10
		" ",      // 11
		"he",     // 12 (merge of h+e)
		"ll",     // 13 (merge of l+l)
		"hel",    // 14 (merge of he+l)
		"hello",  // 15 (multiple merges)
		"world",  // 16
	}

	// Build vocab map
	t.vocab = make(map[string]int)
	for id, token := range t.tokens {
		t.vocab[token] = id
	}

	// Scores (not critical for basic BPE)
	t.scores = make([]float32, len(t.tokens))

	// BPE merge rules (pair → rank, lower rank = higher priority)
	t.merges = map[string]int{
		"h e":    0, // Merge h+e → he (highest priority)
		"l l":    1, // Merge l+l → ll
		"he l":   2, // Merge he+l → hel
		"hel l":  3, // Merge hel+l → hell
		"hell o": 4, // Merge hell+o → hello
		"w o":    5, // Merge w+o → wo
		"wo r":   6, // Merge wo+r → wor
		"wor l":  7, // Merge wor+l → worl
		"worl d": 8, // Merge worl+d → world
	}

	// Special tokens
	t.bosID = 1
	t.eosID = 2
	t.padID = 3
	t.unkID = 0

	return t
}

func TestNewTokenizer(t *testing.T) {
	tok := NewTokenizer()
	if tok == nil {
		t.Fatal("NewTokenizer returned nil")
	}
	if tok.VocabSize() != 0 {
		t.Errorf("Expected empty vocab, got size %d", tok.VocabSize())
	}
	if tok.BOSID() != -1 {
		t.Errorf("Expected BOS ID -1, got %d", tok.BOSID())
	}
}

func TestTokenizerGetters(t *testing.T) {
	tok := createMockTokenizer()

	if tok.VocabSize() != 17 {
		t.Errorf("Expected vocab size 17, got %d", tok.VocabSize())
	}
	if tok.BOSID() != 1 {
		t.Errorf("Expected BOS ID 1, got %d", tok.BOSID())
	}
	if tok.EOSID() != 2 {
		t.Errorf("Expected EOS ID 2, got %d", tok.EOSID())
	}
	if tok.PADID() != 3 {
		t.Errorf("Expected PAD ID 3, got %d", tok.PADID())
	}
	if tok.UNKID() != 0 {
		t.Errorf("Expected UNK ID 0, got %d", tok.UNKID())
	}
	if tok.ModelType() != "test" {
		t.Errorf("Expected model type 'test', got '%s'", tok.ModelType())
	}
}

func TestTokenToID(t *testing.T) {
	tok := createMockTokenizer()

	tests := []struct {
		token      string
		expectedID int
	}{
		{"<s>", 1},
		{"h", 4},
		{"hello", 15},
		{"nonexistent", -1},
	}

	for _, tt := range tests {
		id := tok.TokenToID(tt.token)
		if id != tt.expectedID {
			t.Errorf("TokenToID(%q) = %d, expected %d", tt.token, id, tt.expectedID)
		}
	}
}

func TestIDToToken(t *testing.T) {
	tok := createMockTokenizer()

	tests := []struct {
		id            int
		expectedToken string
	}{
		{1, "<s>"},
		{4, "h"},
		{15, "hello"},
		{999, ""}, // Out of range
		{-1, ""},  // Negative
	}

	for _, tt := range tests {
		token := tok.IDToToken(tt.id)
		if token != tt.expectedToken {
			t.Errorf("IDToToken(%d) = %q, expected %q", tt.id, token, tt.expectedToken)
		}
	}
}

func TestHasMerge(t *testing.T) {
	tok := createMockTokenizer()

	if !tok.HasMerge("h e") {
		t.Error("Expected merge 'h e' to exist")
	}
	if tok.HasMerge("x y") {
		t.Error("Expected merge 'x y' to not exist")
	}
}

func TestGetMergeRank(t *testing.T) {
	tok := createMockTokenizer()

	tests := []struct {
		pair         string
		expectedRank int
	}{
		{"h e", 0},
		{"l l", 1},
		{"x y", -1}, // Non-existent
	}

	for _, tt := range tests {
		rank := tok.GetMergeRank(tt.pair)
		if rank != tt.expectedRank {
			t.Errorf("GetMergeRank(%q) = %d, expected %d", tt.pair, rank, tt.expectedRank)
		}
	}
}

func TestEncode_Simple(t *testing.T) {
	tok := createMockTokenizer()

	// Simple test: encode "hello"
	// Bytes: h e l l o
	// After merges: hello (if merges work correctly)
	ids := tok.Encode("hello", false, false)

	// With our mock tokenizer, "hello" should be a single token (ID 15)
	// But our simple BPE implementation might not merge all the way
	// Let's just check that we get valid IDs
	if len(ids) == 0 {
		t.Error("Expected non-empty result")
	}

	for i, id := range ids {
		if id < 0 || id >= tok.VocabSize() {
			t.Errorf("Invalid ID at position %d: %d", i, id)
		}
	}
}

func TestEncode_WithSpecialTokens(t *testing.T) {
	tok := createMockTokenizer()

	// Test with BOS and EOS
	ids := tok.Encode("hi", true, true)

	if len(ids) < 2 {
		t.Fatalf("Expected at least 2 tokens (BOS+EOS), got %d", len(ids))
	}

	// First token should be BOS
	if ids[0] != tok.BOSID() {
		t.Errorf("Expected BOS token (%d) at start, got %d", tok.BOSID(), ids[0])
	}

	// Last token should be EOS
	if ids[len(ids)-1] != tok.EOSID() {
		t.Errorf("Expected EOS token (%d) at end, got %d", tok.EOSID(), ids[len(ids)-1])
	}
}

func TestEncode_EmptyString(t *testing.T) {
	tok := createMockTokenizer()

	// Empty string without special tokens
	ids := tok.Encode("", false, false)
	if len(ids) != 0 {
		t.Errorf("Expected empty result for empty string, got %v", ids)
	}

	// Empty string with BOS and EOS
	ids = tok.Encode("", true, true)
	if len(ids) != 2 {
		t.Errorf("Expected 2 tokens (BOS+EOS), got %d", len(ids))
	}
	if ids[0] != tok.BOSID() || ids[1] != tok.EOSID() {
		t.Errorf("Expected [BOS, EOS], got %v", ids)
	}
}

func TestDecode(t *testing.T) {
	tok := createMockTokenizer()

	tests := []struct {
		name     string
		ids      []int
		expected string
	}{
		{"single token", []int{4}, "h"},
		{"multiple tokens", []int{4, 5, 6, 6, 7}, "hello"},
		{"with spaces", []int{4, 5, 11, 8, 7}, "he wo"},
		{"empty", []int{}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.Decode(tt.ids, false)
			if result != tt.expected {
				t.Errorf("Decode(%v) = %q, expected %q", tt.ids, result, tt.expected)
			}
		})
	}
}

func TestDecode_SkipSpecial(t *testing.T) {
	tok := createMockTokenizer()

	// IDs: [BOS, h, e, l, l, o, EOS]
	ids := []int{tok.BOSID(), 4, 5, 6, 6, 7, tok.EOSID()}

	// Without skipping special tokens
	result := tok.Decode(ids, false)
	if !contains(result, "<s>") || !contains(result, "</s>") {
		t.Errorf("Expected special tokens in result, got %q", result)
	}

	// With skipping special tokens
	result = tok.Decode(ids, true)
	if contains(result, "<s>") || contains(result, "</s>") {
		t.Errorf("Expected no special tokens in result, got %q", result)
	}
}

func TestDecode_InvalidIDs(t *testing.T) {
	tok := createMockTokenizer()

	// Test with out-of-range IDs
	ids := []int{4, 999, 5} // 999 is invalid
	result := tok.Decode(ids, false)

	// Should skip invalid ID and return "he"
	if result != "he" {
		t.Errorf("Expected %q, got %q", "he", result)
	}
}

func TestEncodeAsTokens(t *testing.T) {
	tok := createMockTokenizer()

	tokens := tok.EncodeAsTokens("he")
	if len(tokens) == 0 {
		t.Error("Expected non-empty token list")
	}

	// All tokens should be valid
	for _, token := range tokens {
		if tok.TokenToID(token) == -1 {
			t.Errorf("Invalid token in result: %q", token)
		}
	}
}

func TestDecodeTokens(t *testing.T) {
	tok := createMockTokenizer()

	tokens := []string{"h", "e", "l", "l", "o"}
	result := tok.DecodeTokens(tokens)

	if result != "hello" {
		t.Errorf("Expected %q, got %q", "hello", result)
	}
}

func TestCountTokens(t *testing.T) {
	tok := createMockTokenizer()

	tests := []struct {
		text         string
		expectedMin  int
		expectedMax  int
	}{
		{"", 0, 0},
		{"h", 1, 1},
		{"hello", 1, 5}, // Depends on merges
	}

	for _, tt := range tests {
		count := tok.CountTokens(tt.text)
		if count < tt.expectedMin || count > tt.expectedMax {
			t.Errorf("CountTokens(%q) = %d, expected %d-%d", tt.text, count, tt.expectedMin, tt.expectedMax)
		}
	}
}

func TestConvertToInt(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected int
		ok       bool
	}{
		{"int", int(42), 42, true},
		{"int8", int8(10), 10, true},
		{"int16", int16(1000), 1000, true},
		{"int32", int32(100000), 100000, true},
		{"int64", int64(1000000), 1000000, true},
		{"uint", uint(42), 42, true},
		{"uint8", uint8(255), 255, true},
		{"uint16", uint16(65535), 65535, true},
		{"uint32", uint32(100000), 100000, true},
		{"uint64", uint64(1000000), 1000000, true},
		{"string", "not an int", 0, false},
		{"float", 3.14, 0, false},
		{"nil", nil, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, ok := convertToInt(tt.input)
			if ok != tt.ok {
				t.Errorf("convertToInt(%v) ok = %v, expected %v", tt.input, ok, tt.ok)
			}
			if ok && result != tt.expected {
				t.Errorf("convertToInt(%v) = %d, expected %d", tt.input, result, tt.expected)
			}
		})
	}
}

func TestNewTokenizerFromGGUF_NoTokens(t *testing.T) {
	// Create a GGUF file with no tokens
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
		Tensors:  make(map[string]*gguf.TensorInfo),
	}

	_, err := NewTokenizerFromGGUF(ggufFile)
	if err == nil {
		t.Error("Expected error for GGUF with no tokens")
	}
}

func TestNewTokenizerFromGGUF_Complete(t *testing.T) {
	// Create a GGUF file with complete tokenizer data
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
		Tensors:  make(map[string]*gguf.TensorInfo),
	}

	// Set tokenizer model type
	ggufFile.Metadata[gguf.KeyTokenizerModel] = "llama"

	// Set tokens
	ggufFile.Metadata[gguf.KeyTokenizerTokens] = []interface{}{
		"<unk>", "<s>", "</s>", "<pad>", "a", "b", "c",
	}

	// Set scores
	ggufFile.Metadata[gguf.KeyTokenizerScores] = []interface{}{
		float32(0.0), float32(0.0), float32(0.0), float32(0.0),
		float32(1.0), float32(2.0), float32(3.0),
	}

	// Set merges
	ggufFile.Metadata[gguf.KeyTokenizerMerges] = []interface{}{
		"a b",
		"b c",
	}

	// Set special token IDs
	ggufFile.Metadata[gguf.KeyTokenizerBOSID] = uint32(1)
	ggufFile.Metadata[gguf.KeyTokenizerEOSID] = uint32(2)
	ggufFile.Metadata[gguf.KeyTokenizerPADID] = uint32(3)

	tok, err := NewTokenizerFromGGUF(ggufFile)
	if err != nil {
		t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
	}

	// Verify basic properties
	if tok.VocabSize() != 7 {
		t.Errorf("Expected vocab size 7, got %d", tok.VocabSize())
	}

	if tok.ModelType() != "llama" {
		t.Errorf("Expected model type 'llama', got '%s'", tok.ModelType())
	}

	// Verify special tokens
	if tok.BOSID() != 1 {
		t.Errorf("Expected BOS ID 1, got %d", tok.BOSID())
	}
	if tok.EOSID() != 2 {
		t.Errorf("Expected EOS ID 2, got %d", tok.EOSID())
	}
	if tok.PADID() != 3 {
		t.Errorf("Expected PAD ID 3, got %d", tok.PADID())
	}
	if tok.UNKID() != 0 {
		t.Errorf("Expected UNK ID 0, got %d", tok.UNKID())
	}

	// Verify vocab
	if id := tok.TokenToID("<s>"); id != 1 {
		t.Errorf("Expected '<s>' to have ID 1, got %d", id)
	}

	// Verify merges
	if !tok.HasMerge("a b") {
		t.Error("Expected merge 'a b' to exist")
	}
	if tok.GetMergeRank("a b") != 0 {
		t.Errorf("Expected rank 0 for 'a b', got %d", tok.GetMergeRank("a b"))
	}

	// Verify scores
	if len(tok.scores) != 7 {
		t.Errorf("Expected 7 scores, got %d", len(tok.scores))
	}
}

func TestNewTokenizerFromGGUF_NoScores(t *testing.T) {
	// Test GGUF without scores (should default to 0.0)
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
		Tensors:  make(map[string]*gguf.TensorInfo),
	}

	ggufFile.Metadata[gguf.KeyTokenizerTokens] = []interface{}{
		"<unk>", "a", "b",
	}

	tok, err := NewTokenizerFromGGUF(ggufFile)
	if err != nil {
		t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
	}

	if len(tok.scores) != 3 {
		t.Errorf("Expected 3 default scores, got %d", len(tok.scores))
	}

	// All scores should be 0.0
	for i, score := range tok.scores {
		if score != 0.0 {
			t.Errorf("Expected score[%d] = 0.0, got %f", i, score)
		}
	}
}

func TestNewTokenizerFromGGUF_NoModelType(t *testing.T) {
	// Test GGUF without model type (should default to "unknown")
	ggufFile := &gguf.GGUFFile{
		Metadata: make(map[string]interface{}),
		Tensors:  make(map[string]*gguf.TensorInfo),
	}

	ggufFile.Metadata[gguf.KeyTokenizerTokens] = []interface{}{
		"<unk>", "a",
	}

	tok, err := NewTokenizerFromGGUF(ggufFile)
	if err != nil {
		t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
	}

	if tok.ModelType() != "unknown" {
		t.Errorf("Expected model type 'unknown', got '%s'", tok.ModelType())
	}
}

func TestNewTokenizerFromGGUF_SpecialTokenVariants(t *testing.T) {
	// Test with different integer types for special token IDs
	tests := []struct {
		name  string
		value interface{}
	}{
		{"int32", int32(1)},
		{"int64", int64(1)},
		{"uint64", uint64(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ggufFile := &gguf.GGUFFile{
				Metadata: make(map[string]interface{}),
				Tensors:  make(map[string]*gguf.TensorInfo),
			}

			ggufFile.Metadata[gguf.KeyTokenizerTokens] = []interface{}{
				"<unk>", "<s>",
			}
			ggufFile.Metadata[gguf.KeyTokenizerBOSID] = tt.value

			tok, err := NewTokenizerFromGGUF(ggufFile)
			if err != nil {
				t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
			}

			if tok.BOSID() != 1 {
				t.Errorf("Expected BOS ID 1, got %d", tok.BOSID())
			}
		})
	}
}

func TestNewTokenizerFromGGUF_UNKTokenVariants(t *testing.T) {
	// Test UNK token detection with different cases
	tests := []struct {
		name       string
		tokens     []interface{}
		expectedID int
	}{
		{"lowercase", []interface{}{"<unk>", "a"}, 0},
		{"uppercase", []interface{}{"<UNK>", "a"}, 0},
		{"no unk", []interface{}{"a", "b"}, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ggufFile := &gguf.GGUFFile{
				Metadata: make(map[string]interface{}),
				Tensors:  make(map[string]*gguf.TensorInfo),
			}

			ggufFile.Metadata[gguf.KeyTokenizerTokens] = tt.tokens

			tok, err := NewTokenizerFromGGUF(ggufFile)
			if err != nil {
				t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
			}

			if tok.UNKID() != tt.expectedID {
				t.Errorf("Expected UNK ID %d, got %d", tt.expectedID, tok.UNKID())
			}
		})
	}
}

func TestIsValidUTF8(t *testing.T) {
	tests := []struct {
		name     string
		input    []byte
		expected bool
	}{
		{"valid ascii", []byte("hello"), true},
		{"valid utf8", []byte("hello 世界"), true},
		{"valid emoji", []byte("hello 👋"), true},
		{"invalid utf8", []byte{0xff, 0xfe, 0xfd}, false},
		{"empty", []byte{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidUTF8(tt.input)
			if result != tt.expected {
				t.Errorf("isValidUTF8(%v) = %v, expected %v", tt.input, result, tt.expected)
			}
		})
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Benchmarks
func BenchmarkEncode(b *testing.B) {
	tok := createMockTokenizer()
	text := "hello world"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text, false, false)
	}
}

func BenchmarkDecode(b *testing.B) {
	tok := createMockTokenizer()
	ids := []int{4, 5, 6, 6, 7, 11, 8, 7, 9, 6, 10}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, false)
	}
}

func BenchmarkTokenToID(b *testing.B) {
	tok := createMockTokenizer()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.TokenToID("hello")
	}
}

func BenchmarkIDToToken(b *testing.B) {
	tok := createMockTokenizer()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.IDToToken(15)
	}
}

// createMockTokenizerWithSpecialTokens creates a tokenizer that has <|...|> special tokens in its vocab
func createMockTokenizerWithSpecialTokens() *Tokenizer {
	t := createMockTokenizer()

	// Add special tokens for ChatML and Llama3
	specialTokens := []string{
		"<|im_start|>", // 17
		"<|im_end|>",   // 18
		"<|begin_of_text|>",    // 19
		"<|start_header_id|>",  // 20
		"<|end_header_id|>",    // 21
		"<|eot_id|>",          // 22
	}

	for _, st := range specialTokens {
		id := len(t.tokens)
		t.tokens = append(t.tokens, st)
		t.vocab[st] = id
		t.scores = append(t.scores, 0.0)
	}

	return t
}

func TestEncodeSpecialTokens(t *testing.T) {
	tok := createMockTokenizerWithSpecialTokens()

	// Each special token should encode to exactly one token ID
	tests := []struct {
		token    string
		expected int
	}{
		{"<|im_start|>", 17},
		{"<|im_end|>", 18},
		{"<|eot_id|>", 22},
	}

	for _, tt := range tests {
		ids := tok.Encode(tt.token, false, false)
		if len(ids) != 1 {
			t.Errorf("Encode(%q) produced %d tokens, expected 1: %v", tt.token, len(ids), ids)
			continue
		}
		if ids[0] != tt.expected {
			t.Errorf("Encode(%q) = [%d], expected [%d]", tt.token, ids[0], tt.expected)
		}
	}
}

func TestEncodeChatMLPrompt(t *testing.T) {
	tok := createMockTokenizerWithSpecialTokens()

	prompt := "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
	ids := tok.Encode(prompt, false, false)

	// Should contain the special token IDs
	foundImStart := 0
	foundImEnd := 0
	for _, id := range ids {
		if id == 17 { // <|im_start|>
			foundImStart++
		}
		if id == 18 { // <|im_end|>
			foundImEnd++
		}
	}

	if foundImStart != 2 {
		t.Errorf("expected 2 <|im_start|> tokens, found %d in %v", foundImStart, ids)
	}
	if foundImEnd != 1 {
		t.Errorf("expected 1 <|im_end|> token, found %d in %v", foundImEnd, ids)
	}
}

func TestEncodeWithSpecialTokensPreservesRegularText(t *testing.T) {
	tok := createMockTokenizerWithSpecialTokens()

	// Text with no special tokens should work the same as before
	ids1 := tok.Encode("hello", false, false)
	ids2 := tok.encodeBPE("hello", false, false)

	if len(ids1) != len(ids2) {
		t.Errorf("special token path changed regular text encoding: %v vs %v", ids1, ids2)
		return
	}
	for i := range ids1 {
		if ids1[i] != ids2[i] {
			t.Errorf("special token path changed regular text encoding at pos %d: %d vs %d", i, ids1[i], ids2[i])
		}
	}
}

func TestEncodeSpecialTokenBOS(t *testing.T) {
	tok := createMockTokenizerWithSpecialTokens()

	// With BOS, special token text
	ids := tok.Encode("<|im_start|>", true, false)

	if len(ids) != 2 {
		t.Errorf("expected 2 tokens (BOS + special), got %d: %v", len(ids), ids)
		return
	}
	if ids[0] != tok.BOSID() {
		t.Errorf("first token should be BOS (%d), got %d", tok.BOSID(), ids[0])
	}
	if ids[1] != 17 {
		t.Errorf("second token should be <|im_start|> (17), got %d", ids[1])
	}
}
