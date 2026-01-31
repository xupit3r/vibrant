package model

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestNewCacheManager(t *testing.T) {
	tmpDir := t.TempDir()
	
	cm, err := NewCacheManager(tmpDir, 10)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}
	
	if cm.CacheDir != tmpDir {
		t.Errorf("Expected cache dir %s, got %s", tmpDir, cm.CacheDir)
	}
	
	if cm.MaxCacheSizeGB != 10 {
		t.Errorf("Expected max size 10GB, got %d", cm.MaxCacheSizeGB)
	}
	
	// Check manifest was created
	manifestPath := filepath.Join(tmpDir, "manifest.json")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		t.Error("Manifest file was not created")
	}
}

func TestCacheAddAndGet(t *testing.T) {
	tmpDir := t.TempDir()
	cm, _ := NewCacheManager(tmpDir, 10)
	
	// Create a test file
	testFile := filepath.Join(tmpDir, "test-model.gguf")
	content := []byte("test model data")
	if err := os.WriteFile(testFile, content, 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	
	// Add to cache
	err := cm.Add("test-model", testFile, "abc123")
	if err != nil {
		t.Fatalf("Failed to add model to cache: %v", err)
	}
	
	// Verify it's in the cache
	if !cm.Has("test-model") {
		t.Error("Model should be in cache")
	}
	
	// Get from cache
	cached, err := cm.Get("test-model")
	if err != nil {
		t.Fatalf("Failed to get model from cache: %v", err)
	}
	
	if cached.ID != "test-model" {
		t.Errorf("Expected ID test-model, got %s", cached.ID)
	}
	
	if cached.Checksum != "abc123" {
		t.Errorf("Expected checksum abc123, got %s", cached.Checksum)
	}
	
	if cached.SizeBytes != int64(len(content)) {
		t.Errorf("Expected size %d, got %d", len(content), cached.SizeBytes)
	}
}

func TestCacheRemove(t *testing.T) {
	tmpDir := t.TempDir()
	cm, _ := NewCacheManager(tmpDir, 10)
	
	// Create and add test file
	testFile := filepath.Join(tmpDir, "test-model.gguf")
	os.WriteFile(testFile, []byte("test"), 0644)
	cm.Add("test-model", testFile, "")
	
	// Remove
	err := cm.Remove("test-model")
	if err != nil {
		t.Fatalf("Failed to remove model: %v", err)
	}
	
	// Verify removed
	if cm.Has("test-model") {
		t.Error("Model should not be in cache after removal")
	}
	
	// Verify file deleted
	if _, err := os.Stat(testFile); !os.IsNotExist(err) {
		t.Error("Model file should be deleted")
	}
}

func TestCacheList(t *testing.T) {
	tmpDir := t.TempDir()
	cm, _ := NewCacheManager(tmpDir, 10)
	
	// Initially empty
	if len(cm.List()) != 0 {
		t.Error("Expected empty cache")
	}
	
	// Add models
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("test-model-%d", i)
		testFile := filepath.Join(tmpDir, id+".gguf")
		os.WriteFile(testFile, []byte("test"), 0644)
		cm.Add(id, testFile, "")
	}
	
	models := cm.List()
	if len(models) != 3 {
		t.Errorf("Expected 3 models, got %d", len(models))
	}
}

func TestCacheGetTotalSize(t *testing.T) {
	tmpDir := t.TempDir()
	cm, _ := NewCacheManager(tmpDir, 10)
	
	// Add files of known sizes
	sizes := []int{100, 200, 300}
	for i, size := range sizes {
		id := fmt.Sprintf("test-model-%d", i)
		testFile := filepath.Join(tmpDir, id)
		os.WriteFile(testFile, make([]byte, size), 0644)
		cm.Add(id, testFile, "")
	}
	
	totalSize := cm.GetTotalSize()
	expectedSize := int64(100 + 200 + 300)
	if totalSize != expectedSize {
		t.Errorf("Expected total size %d, got %d", expectedSize, totalSize)
	}
}

func TestCacheUpdateLastUsed(t *testing.T) {
	tmpDir := t.TempDir()
	cm, _ := NewCacheManager(tmpDir, 10)
	
	// Add model
	testFile := filepath.Join(tmpDir, "test-model.gguf")
	os.WriteFile(testFile, []byte("test"), 0644)
	cm.Add("test-model", testFile, "")
	
	// Get initial last used time
	cached, _ := cm.Get("test-model")
	initialTime := cached.LastUsed
	initialCount := cached.UseCount
	
	// Update last used
	err := cm.UpdateLastUsed("test-model")
	if err != nil {
		t.Fatalf("Failed to update last used: %v", err)
	}
	
	// Verify updated
	cached, _ = cm.Get("test-model")
	if !cached.LastUsed.After(initialTime) {
		t.Error("Last used time should be updated")
	}
	
	if cached.UseCount != initialCount+1 {
		t.Errorf("Expected use count %d, got %d", initialCount+1, cached.UseCount)
	}
}

func TestComputeSHA256(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	
	content := []byte("test content")
	os.WriteFile(testFile, content, 0644)
	
	checksum, err := ComputeSHA256(testFile)
	if err != nil {
		t.Fatalf("Failed to compute checksum: %v", err)
	}
	
	if checksum == "" {
		t.Error("Expected non-empty checksum")
	}
	
	if len(checksum) != 64 { // SHA256 is 64 hex chars
		t.Errorf("Expected 64 character checksum, got %d", len(checksum))
	}
}
