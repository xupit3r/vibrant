package model

import (
	"fmt"
	
	"github.com/xupit3r/vibrant/internal/system"
)

// Selector handles model selection based on system resources
type Selector struct {
	usableRAM int64
}

// NewSelector creates a new model selector
func NewSelector() (*Selector, error) {
	usableRAM, err := system.EstimateUsableRAM()
	if err != nil {
		return nil, fmt.Errorf("failed to determine usable RAM: %w", err)
	}
	
	return &Selector{
		usableRAM: usableRAM,
	}, nil
}

// SelectBest automatically selects the best model for the system
func (s *Selector) SelectBest() (*ModelInfo, error) {
	// Filter models that fit in RAM
	candidates := FilterByRAM(s.usableRAM)
	if len(candidates) == 0 {
		ramGB := float64(s.usableRAM) / (1024 * 1024 * 1024)
		return nil, fmt.Errorf("insufficient RAM (%.1f GB available, need at least 4 GB)", ramGB)
	}
	
	// Prefer recommended models
	recommended := FilterRecommended(candidates)
	if len(recommended) > 0 {
		candidates = recommended
	}
	
	// Sort by RAM requirement (descending) and select the largest that fits
	sorted := SortByRAMRequired(candidates)
	
	return &sorted[0], nil
}

// SelectWithPreference selects a model with user preferences
func (s *Selector) SelectWithPreference(preferFamily string, preferQuantization string) (*ModelInfo, error) {
	candidates := FilterByRAM(s.usableRAM)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("insufficient RAM for any model")
	}
	
	// Filter by family if specified
	if preferFamily != "" {
		var familyMatch []ModelInfo
		for _, model := range candidates {
			if model.Family == preferFamily {
				familyMatch = append(familyMatch, model)
			}
		}
		if len(familyMatch) > 0 {
			candidates = familyMatch
		}
	}
	
	// Filter by quantization if specified
	if preferQuantization != "" {
		var quantMatch []ModelInfo
		for _, model := range candidates {
			if model.Quantization == preferQuantization {
				quantMatch = append(quantMatch, model)
			}
		}
		if len(quantMatch) > 0 {
			candidates = quantMatch
		}
	}
	
	// Prefer recommended, then largest
	recommended := FilterRecommended(candidates)
	if len(recommended) > 0 {
		candidates = recommended
	}
	
	sorted := SortByRAMRequired(candidates)
	return &sorted[0], nil
}

// CanFit checks if a specific model can fit in available RAM
func (s *Selector) CanFit(modelID string) (bool, error) {
	model, err := GetModelByID(modelID)
	if err != nil {
		return false, err
	}
	
	requiredBytes := int64(model.RAMRequiredMB) * 1024 * 1024
	return requiredBytes <= s.usableRAM, nil
}

// GetUsableRAM returns the usable RAM in bytes
func (s *Selector) GetUsableRAM() int64 {
	return s.usableRAM
}

// GetRecommendedTier returns a human-readable recommendation tier
func (s *Selector) GetRecommendedTier() string {
	ramGB := float64(s.usableRAM) / (1024 * 1024 * 1024)
	
	switch {
	case ramGB < 6:
		return "Insufficient RAM for any model"
	case ramGB < 10:
		return "3B models (Q4_K_M)"
	case ramGB < 16:
		return "7B models (Q5_K_M)"
	case ramGB < 24:
		return "14B models (Q5_K_M)"
	default:
		return "14B+ models (Q8_0 or larger)"
	}
}
