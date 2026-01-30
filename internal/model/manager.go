package model

import (
	"fmt"
)

// Manager coordinates model operations
type Manager struct {
	Cache      *CacheManager
	Selector   *Selector
	Downloader *Downloader
}

// NewManager creates a new model manager
func NewManager(cacheDir string, maxCacheSizeGB int) (*Manager, error) {
	cache, err := NewCacheManager(cacheDir, maxCacheSizeGB)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache manager: %w", err)
	}
	
	selector, err := NewSelector()
	if err != nil {
		return nil, fmt.Errorf("failed to create selector: %w", err)
	}
	
	downloader := NewDownloader(cache)
	
	return &Manager{
		Cache:      cache,
		Selector:   selector,
		Downloader: downloader,
	}, nil
}

// EnsureModel ensures a model is available (downloads if needed)
func (m *Manager) EnsureModel(modelID string) error {
	// Check if model exists in registry
	_, err := GetModelByID(modelID)
	if err != nil {
		return err
	}
	
	// Check if already cached
	if m.Cache.Has(modelID) {
		// Verify checksum
		valid, err := m.Cache.VerifyChecksum(modelID)
		if err != nil {
			return fmt.Errorf("failed to verify model: %w", err)
		}
		if valid {
			return nil
		}
	}
	
	// Download model
	return m.Downloader.Download(modelID)
}

// GetOrSelectModel gets a model by ID or auto-selects the best one
func (m *Manager) GetOrSelectModel(modelID string) (*ModelInfo, error) {
	if modelID == "" || modelID == "auto" {
		return m.Selector.SelectBest()
	}
	
	model, err := GetModelByID(modelID)
	if err != nil {
		return nil, err
	}
	
	// Check if model can fit in RAM
	canFit, err := m.Selector.CanFit(modelID)
	if err != nil {
		return nil, err
	}
	
	if !canFit {
		return nil, fmt.Errorf("model %s requires more RAM than available", modelID)
	}
	
	return model, nil
}
