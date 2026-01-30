package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// ProgressFunc is called during download to report progress
type ProgressFunc func(downloaded, total int64, speed float64)

// Downloader handles model downloads from HuggingFace
type Downloader struct {
	CacheManager *CacheManager
	Client       *http.Client
	ProgressFunc ProgressFunc
}

// NewDownloader creates a new downloader
func NewDownloader(cacheManager *CacheManager) *Downloader {
	return &Downloader{
		CacheManager: cacheManager,
		Client: &http.Client{
			Timeout: 0, // No timeout for large downloads
		},
		ProgressFunc: nil,
	}
}

// Download downloads a model from HuggingFace
func (d *Downloader) Download(modelID string) error {
	// Get model info
	model, err := GetModelByID(modelID)
	if err != nil {
		return err
	}
	
	// Check if already cached and valid
	if d.CacheManager.Has(modelID) {
		valid, err := d.CacheManager.VerifyChecksum(modelID)
		if err == nil && valid {
			// Model already downloaded and valid
			return nil
		}
		// Invalid checksum, re-download
		if err := d.CacheManager.Remove(modelID); err != nil {
			return fmt.Errorf("failed to remove invalid cached model: %w", err)
		}
	}
	
	// Build download URL
	url := BuildHuggingFaceURL(model)
	
	// Get destination paths
	finalPath := d.CacheManager.GetModelPath(modelID)
	tempPath := d.CacheManager.GetDownloadPath(modelID)
	
	// Check for partial download
	var offset int64
	if info, err := os.Stat(tempPath); err == nil {
		offset = info.Size()
	}
	
	// Create HTTP request with range support
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	
	// Execute request
	resp, err := d.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}
	defer resp.Body.Close()
	
	// Check response
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}
	
	// Get total size
	totalSize := resp.ContentLength
	if offset > 0 && resp.StatusCode == http.StatusPartialContent {
		totalSize += offset
	}
	
	// Open/create temp file
	flag := os.O_CREATE | os.O_WRONLY
	if offset > 0 {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	
	file, err := os.OpenFile(tempPath, flag, 0644)
	if err != nil {
		return fmt.Errorf("failed to open temp file: %w", err)
	}
	defer file.Close()
	
	// Download with progress tracking
	if err := d.downloadWithProgress(resp.Body, file, offset, totalSize); err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	
	// Verify checksum if provided
	if model.SHA256 != "" {
		computed, err := ComputeSHA256(tempPath)
		if err != nil {
			return fmt.Errorf("failed to compute checksum: %w", err)
		}
		if computed != model.SHA256 {
			os.Remove(tempPath)
			return fmt.Errorf("checksum mismatch: expected %s, got %s", model.SHA256, computed)
		}
	}
	
	// Move to final location
	if err := os.Rename(tempPath, finalPath); err != nil {
		return fmt.Errorf("failed to move file to final location: %w", err)
	}
	
	// Add to cache manifest
	checksum := model.SHA256
	if checksum == "" {
		// Compute checksum if not provided
		checksum, _ = ComputeSHA256(finalPath)
	}
	
	if err := d.CacheManager.Add(modelID, finalPath, checksum); err != nil {
		return fmt.Errorf("failed to update cache manifest: %w", err)
	}
	
	return nil
}

// downloadWithProgress downloads with progress reporting
func (d *Downloader) downloadWithProgress(src io.Reader, dst io.Writer, offset, total int64) error {
	buf := make([]byte, 32*1024) // 32KB buffer
	downloaded := offset
	startTime := time.Now()
	lastUpdate := time.Now()
	
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)
			
			// Report progress every 500ms
			if d.ProgressFunc != nil && time.Since(lastUpdate) > 500*time.Millisecond {
				elapsed := time.Since(startTime).Seconds()
				speed := float64(downloaded-offset) / elapsed // bytes per second
				d.ProgressFunc(downloaded, total, speed)
				lastUpdate = time.Now()
			}
		}
		
		if err == io.EOF {
			// Final progress update
			if d.ProgressFunc != nil {
				elapsed := time.Since(startTime).Seconds()
				speed := float64(downloaded-offset) / elapsed
				d.ProgressFunc(downloaded, total, speed)
			}
			break
		}
		
		if err != nil {
			return err
		}
	}
	
	return nil
}

// BuildHuggingFaceURL builds the download URL for a model
func BuildHuggingFaceURL(model *ModelInfo) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s",
		model.HuggingFaceRepo,
		model.Filename)
}

// CleanupFailedDownloads removes partial downloads older than 24 hours
func (d *Downloader) CleanupFailedDownloads() error {
	downloadDir := filepath.Join(d.CacheManager.CacheDir, ".downloading")
	
	entries, err := os.ReadDir(downloadDir)
	if err != nil {
		return err
	}
	
	cutoff := time.Now().Add(-24 * time.Hour)
	
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		
		info, err := entry.Info()
		if err != nil {
			continue
		}
		
		if info.ModTime().Before(cutoff) {
			path := filepath.Join(downloadDir, entry.Name())
			os.Remove(path)
		}
	}
	
	return nil
}
