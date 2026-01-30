package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// CachedModel represents a model stored in the cache
type CachedModel struct {
	ID           string    `json:"id"`
	Path         string    `json:"path"`
	SizeBytes    int64     `json:"size_bytes"`
	Checksum     string    `json:"checksum"`
	DownloadedAt time.Time `json:"downloaded_at"`
	LastUsed     time.Time `json:"last_used"`
	UseCount     int       `json:"use_count"`
}

// CacheManifest tracks cached models
type CacheManifest struct {
	Version string         `json:"version"`
	Models  []CachedModel  `json:"models"`
}

// CacheManager manages the model cache
type CacheManager struct {
	CacheDir         string
	MaxCacheSizeGB   int
	manifest         *CacheManifest
	manifestPath     string
}

// NewCacheManager creates a new cache manager
func NewCacheManager(cacheDir string, maxCacheSizeGB int) (*CacheManager, error) {
	// Expand home directory
	if len(cacheDir) > 0 && cacheDir[0] == '~' {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get home directory: %w", err)
		}
		cacheDir = filepath.Join(home, cacheDir[1:])
	}
	
	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}
	
	// Create .downloading subdirectory
	downloadDir := filepath.Join(cacheDir, ".downloading")
	if err := os.MkdirAll(downloadDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create download directory: %w", err)
	}
	
	cm := &CacheManager{
		CacheDir:       cacheDir,
		MaxCacheSizeGB: maxCacheSizeGB,
		manifestPath:   filepath.Join(cacheDir, "manifest.json"),
	}
	
	// Load or create manifest
	if err := cm.loadManifest(); err != nil {
		return nil, err
	}
	
	return cm, nil
}

// loadManifest loads the cache manifest from disk
func (cm *CacheManager) loadManifest() error {
	data, err := os.ReadFile(cm.manifestPath)
	if os.IsNotExist(err) {
		// Create new manifest
		cm.manifest = &CacheManifest{
			Version: "1.0",
			Models:  []CachedModel{},
		}
		return cm.saveManifest()
	}
	if err != nil {
		return fmt.Errorf("failed to read manifest: %w", err)
	}
	
	var manifest CacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("failed to parse manifest: %w", err)
	}
	
	cm.manifest = &manifest
	return nil
}

// saveManifest saves the cache manifest to disk
func (cm *CacheManager) saveManifest() error {
	data, err := json.MarshalIndent(cm.manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	
	if err := os.WriteFile(cm.manifestPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}
	
	return nil
}

// List returns all cached models
func (cm *CacheManager) List() []CachedModel {
	return cm.manifest.Models
}

// Get returns a cached model by ID
func (cm *CacheManager) Get(modelID string) (*CachedModel, error) {
	for i := range cm.manifest.Models {
		if cm.manifest.Models[i].ID == modelID {
			return &cm.manifest.Models[i], nil
		}
	}
	return nil, fmt.Errorf("model not found in cache: %s", modelID)
}

// Has checks if a model is cached
func (cm *CacheManager) Has(modelID string) bool {
	_, err := cm.Get(modelID)
	return err == nil
}

// Add adds a model to the cache
func (cm *CacheManager) Add(modelID string, path string, checksum string) error {
	// Get file info
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("failed to stat model file: %w", err)
	}
	
	// Check if already exists
	for i := range cm.manifest.Models {
		if cm.manifest.Models[i].ID == modelID {
			// Update existing entry
			cm.manifest.Models[i].Path = path
			cm.manifest.Models[i].SizeBytes = info.Size()
			cm.manifest.Models[i].Checksum = checksum
			cm.manifest.Models[i].LastUsed = time.Now()
			return cm.saveManifest()
		}
	}
	
	// Add new entry
	cached := CachedModel{
		ID:           modelID,
		Path:         path,
		SizeBytes:    info.Size(),
		Checksum:     checksum,
		DownloadedAt: time.Now(),
		LastUsed:     time.Now(),
		UseCount:     0,
	}
	
	cm.manifest.Models = append(cm.manifest.Models, cached)
	return cm.saveManifest()
}

// Remove removes a model from the cache
func (cm *CacheManager) Remove(modelID string) error {
	cached, err := cm.Get(modelID)
	if err != nil {
		return err
	}
	
	// Delete file
	if err := os.Remove(cached.Path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete model file: %w", err)
	}
	
	// Remove from manifest
	for i := range cm.manifest.Models {
		if cm.manifest.Models[i].ID == modelID {
			cm.manifest.Models = append(cm.manifest.Models[:i], cm.manifest.Models[i+1:]...)
			break
		}
	}
	
	return cm.saveManifest()
}

// UpdateLastUsed updates the last used timestamp for a model
func (cm *CacheManager) UpdateLastUsed(modelID string) error {
	for i := range cm.manifest.Models {
		if cm.manifest.Models[i].ID == modelID {
			cm.manifest.Models[i].LastUsed = time.Now()
			cm.manifest.Models[i].UseCount++
			return cm.saveManifest()
		}
	}
	return fmt.Errorf("model not found in cache: %s", modelID)
}

// GetTotalSize returns the total size of cached models in bytes
func (cm *CacheManager) GetTotalSize() int64 {
	var total int64
	for _, model := range cm.manifest.Models {
		total += model.SizeBytes
	}
	return total
}

// VerifyChecksum verifies the checksum of a cached model
func (cm *CacheManager) VerifyChecksum(modelID string) (bool, error) {
	cached, err := cm.Get(modelID)
	if err != nil {
		return false, err
	}
	
	if cached.Checksum == "" {
		// No checksum to verify
		return true, nil
	}
	
	computed, err := ComputeSHA256(cached.Path)
	if err != nil {
		return false, err
	}
	
	return computed == cached.Checksum, nil
}

// Prune removes least recently used models to stay under size limit
func (cm *CacheManager) Prune() error {
	if cm.MaxCacheSizeGB <= 0 {
		return nil // No limit
	}
	
	maxBytes := int64(cm.MaxCacheSizeGB) * 1024 * 1024 * 1024
	totalSize := cm.GetTotalSize()
	
	if totalSize <= maxBytes {
		return nil // Under limit
	}
	
	// Sort by last used (oldest first)
	models := make([]CachedModel, len(cm.manifest.Models))
	copy(models, cm.manifest.Models)
	
	// Remove models until under limit
	for totalSize > maxBytes && len(models) > 0 {
		// Find least recently used
		oldestIdx := 0
		for i := range models {
			if models[i].LastUsed.Before(models[oldestIdx].LastUsed) {
				oldestIdx = i
			}
		}
		
		// Remove it
		if err := cm.Remove(models[oldestIdx].ID); err != nil {
			return err
		}
		
		totalSize -= models[oldestIdx].SizeBytes
		models = append(models[:oldestIdx], models[oldestIdx+1:]...)
	}
	
	return nil
}

// Clear removes all cached models
func (cm *CacheManager) Clear() error {
	for _, model := range cm.manifest.Models {
		if err := os.Remove(model.Path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to delete %s: %w", model.Path, err)
		}
	}
	
	cm.manifest.Models = []CachedModel{}
	return cm.saveManifest()
}

// GetModelPath returns the full path for a model file
func (cm *CacheManager) GetModelPath(modelID string) string {
	return filepath.Join(cm.CacheDir, modelID+".gguf")
}

// GetDownloadPath returns the temporary download path for a model
func (cm *CacheManager) GetDownloadPath(modelID string) string {
	return filepath.Join(cm.CacheDir, ".downloading", modelID+".gguf.part")
}

// ComputeSHA256 computes the SHA256 checksum of a file
func ComputeSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	
	return hex.EncodeToString(hash.Sum(nil)), nil
}
