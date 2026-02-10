package gguf

import "fmt"

// GetArchitecture returns the model architecture (llama, qwen, mistral, etc.)
func (g *GGUFFile) GetArchitecture() string {
	if arch, ok := g.Metadata[KeyArchitecture].(string); ok {
		return arch
	}
	return "unknown"
}

// GetMetadataString gets a string metadata value with architecture substitution
// If the key contains a %s placeholder, it will be replaced with the architecture name
func (g *GGUFFile) GetMetadataString(key string) (string, bool) {
	// Try direct lookup first
	if val, ok := g.Metadata[key].(string); ok {
		return val, true
	}

	// Try with architecture substitution
	arch := g.GetArchitecture()
	fullKey := fmt.Sprintf(key, arch)
	val, ok := g.Metadata[fullKey].(string)
	return val, ok
}

// GetMetadataInt gets an integer metadata value with architecture substitution
// Handles conversion from various integer types (int8-int64, uint8-uint64)
func (g *GGUFFile) GetMetadataInt(key string) (int, bool) {
	// Try direct lookup first
	if val := convertToInt(g.Metadata[key]); val != nil {
		return *val, true
	}

	// Try with architecture substitution
	arch := g.GetArchitecture()
	fullKey := fmt.Sprintf(key, arch)
	if val := convertToInt(g.Metadata[fullKey]); val != nil {
		return *val, true
	}

	return 0, false
}

// GetMetadataFloat gets a float metadata value with architecture substitution
// Handles conversion from float32 and float64
func (g *GGUFFile) GetMetadataFloat(key string) (float64, bool) {
	// Try direct lookup first
	if val := convertToFloat(g.Metadata[key]); val != nil {
		return *val, true
	}

	// Try with architecture substitution
	arch := g.GetArchitecture()
	fullKey := fmt.Sprintf(key, arch)
	if val := convertToFloat(g.Metadata[fullKey]); val != nil {
		return *val, true
	}

	return 0, false
}

// GetMetadataBool gets a boolean metadata value
func (g *GGUFFile) GetMetadataBool(key string) (bool, bool) {
	// Try direct lookup first
	if val, ok := g.Metadata[key].(bool); ok {
		return val, true
	}

	// Try with architecture substitution
	arch := g.GetArchitecture()
	fullKey := fmt.Sprintf(key, arch)
	val, ok := g.Metadata[fullKey].(bool)
	return val, ok
}

// GetTokens extracts the tokenizer vocabulary
func (g *GGUFFile) GetTokens() []string {
	if tokens, ok := g.Metadata[KeyTokenizerTokens].([]interface{}); ok {
		result := make([]string, 0, len(tokens))
		for _, t := range tokens {
			if s, ok := t.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

// GetTokenScores extracts the tokenizer token scores
func (g *GGUFFile) GetTokenScores() []float32 {
	if scores, ok := g.Metadata[KeyTokenizerScores].([]interface{}); ok {
		result := make([]float32, 0, len(scores))
		for _, s := range scores {
			if f, ok := s.(float32); ok {
				result = append(result, f)
			}
		}
		return result
	}
	return nil
}

// GetMerges extracts the BPE merges
func (g *GGUFFile) GetMerges() []string {
	if merges, ok := g.Metadata[KeyTokenizerMerges].([]interface{}); ok {
		result := make([]string, 0, len(merges))
		for _, m := range merges {
			if s, ok := m.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

// GetChatTemplate returns the raw chat template string from GGUF metadata
func (g *GGUFFile) GetChatTemplate() (string, bool) {
	if tmpl, ok := g.Metadata[KeyChatTemplate].(string); ok {
		return tmpl, true
	}
	return "", false
}

// convertToInt converts various integer types to int
func convertToInt(val interface{}) *int {
	if val == nil {
		return nil
	}

	var result int
	switch v := val.(type) {
	case int:
		result = v
	case int8:
		result = int(v)
	case int16:
		result = int(v)
	case int32:
		result = int(v)
	case int64:
		result = int(v)
	case uint8:
		result = int(v)
	case uint16:
		result = int(v)
	case uint32:
		result = int(v)
	case uint64:
		result = int(v)
	default:
		return nil
	}
	return &result
}

// convertToFloat converts float32 and float64 to float64
func convertToFloat(val interface{}) *float64 {
	if val == nil {
		return nil
	}

	var result float64
	switch v := val.(type) {
	case float32:
		result = float64(v)
	case float64:
		result = v
	default:
		return nil
	}
	return &result
}
