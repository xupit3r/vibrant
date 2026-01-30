package model

import (
	"fmt"
	"sort"
	"strings"
)

// ModelInfo contains metadata about a model
type ModelInfo struct {
	ID              string   // Unique identifier
	Name            string   // Display name
	Family          string   // Model family (qwen, deepseek, codellama)
	Parameters      string   // Size (3B, 7B, 14B)
	Quantization    string   // Q4_K_M, Q5_K_M, Q8_0
	ContextWindow   int      // Max context tokens
	FileSizeMB      int      // Approximate download size
	RAMRequiredMB   int      // Minimum RAM needed
	HuggingFaceRepo string   // HF repository path
	Filename        string   // GGUF filename
	SHA256          string   // Checksum for verification
	Recommended     bool     // Recommended for this size class
	Description     string   // Brief description
	Tags            []string // coding, instruction-following, etc.
}

// Registry contains all available models
var Registry = []ModelInfo{
	// Small models (< 8GB RAM)
	{
		ID:              "qwen2.5-coder-3b-q4",
		Name:            "Qwen 2.5 Coder 3B (Q4_K_M)",
		Family:          "qwen",
		Parameters:      "3B",
		Quantization:    "Q4_K_M",
		ContextWindow:   32768,
		FileSizeMB:      1900,
		RAMRequiredMB:   4000,
		HuggingFaceRepo: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
		Filename:        "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
		SHA256:          "",
		Recommended:     true,
		Description:     "Fast, efficient coding model for systems with limited RAM",
		Tags:            []string{"coding", "fast", "efficient"},
	},
	// Medium models (8-16GB RAM)
	{
		ID:              "qwen2.5-coder-7b-q4",
		Name:            "Qwen 2.5 Coder 7B (Q4_K_M)",
		Family:          "qwen",
		Parameters:      "7B",
		Quantization:    "Q4_K_M",
		ContextWindow:   32768,
		FileSizeMB:      4200,
		RAMRequiredMB:   8000,
		HuggingFaceRepo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
		Filename:        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
		SHA256:          "",
		Recommended:     false,
		Description:     "Good balance of speed and quality",
		Tags:            []string{"coding", "balanced"},
	},
	{
		ID:              "qwen2.5-coder-7b-q5",
		Name:            "Qwen 2.5 Coder 7B (Q5_K_M)",
		Family:          "qwen",
		Parameters:      "7B",
		Quantization:    "Q5_K_M",
		ContextWindow:   32768,
		FileSizeMB:      5100,
		RAMRequiredMB:   10000,
		HuggingFaceRepo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
		Filename:        "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
		SHA256:          "",
		Recommended:     true,
		Description:     "Balanced performance and quality for coding tasks",
		Tags:            []string{"coding", "balanced", "recommended"},
	},
	// Large models (16-32GB RAM)
	{
		ID:              "qwen2.5-coder-14b-q5",
		Name:            "Qwen 2.5 Coder 14B (Q5_K_M)",
		Family:          "qwen",
		Parameters:      "14B",
		Quantization:    "Q5_K_M",
		ContextWindow:   32768,
		FileSizeMB:      9800,
		RAMRequiredMB:   18000,
		HuggingFaceRepo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
		Filename:        "qwen2.5-coder-14b-instruct-q5_k_m.gguf",
		SHA256:          "",
		Recommended:     true,
		Description:     "High-quality coding assistance for systems with ample RAM",
		Tags:            []string{"coding", "high-quality", "large"},
	},
}

// GetModelByID returns a model by its ID
func GetModelByID(id string) (*ModelInfo, error) {
	for _, model := range Registry {
		if model.ID == id {
			return &model, nil
		}
	}
	return nil, fmt.Errorf("model not found: %s", id)
}

// FilterByRAM filters models that can fit in the given RAM (in bytes)
func FilterByRAM(availableBytes int64) []ModelInfo {
	availableMB := availableBytes / (1024 * 1024)
	var filtered []ModelInfo
	
	for _, model := range Registry {
		if int64(model.RAMRequiredMB) <= availableMB {
			filtered = append(filtered, model)
		}
	}
	
	return filtered
}

// FilterByTag filters models by a specific tag
func FilterByTag(models []ModelInfo, tag string) []ModelInfo {
	var filtered []ModelInfo
	
	for _, model := range models {
		for _, t := range model.Tags {
			if strings.EqualFold(t, tag) {
				filtered = append(filtered, model)
				break
			}
		}
	}
	
	return filtered
}

// FilterRecommended returns only recommended models
func FilterRecommended(models []ModelInfo) []ModelInfo {
	var filtered []ModelInfo
	
	for _, model := range models {
		if model.Recommended {
			filtered = append(filtered, model)
		}
	}
	
	return filtered
}

// SortByRAMRequired sorts models by RAM requirement (descending)
func SortByRAMRequired(models []ModelInfo) []ModelInfo {
	sorted := make([]ModelInfo, len(models))
	copy(sorted, models)
	
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].RAMRequiredMB > sorted[j].RAMRequiredMB
	})
	
	return sorted
}

// ListAll returns all models in the registry
func ListAll() []ModelInfo {
	return Registry
}
