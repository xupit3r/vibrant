# Model Management Specification

## Overview
The model management system handles model discovery, selection, downloading, caching, and lifecycle management.

## Model Registry

### Registry Format
Models are defined in an embedded registry with the following schema:

```go
type ModelInfo struct {
    ID              string            // Unique identifier (e.g., "qwen2.5-coder-3b-q4")
    Name            string            // Display name
    Family          string            // Model family (qwen, deepseek, codellama)
    Parameters      string            // Size (3B, 7B, 14B)
    Quantization    string            // Q4_K_M, Q5_K_M, Q8_0
    ContextWindow   int               // Max context tokens
    FileSizeMB      int               // Approximate download size
    RAMRequiredMB   int               // Minimum RAM needed
    HuggingFaceRepo string            // HF repository path
    Filename        string            // GGUF filename
    SHA256          string            // Checksum for verification
    Recommended     bool              // Recommended for this size class
    Description     string            // Brief description
    Tags            []string          // coding, instruction-following, etc.
}
```

### Initial Registry (Phase 2)

```go
var ModelRegistry = []ModelInfo{
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
        SHA256:          "", // To be filled
        Recommended:     true,
        Description:     "Fast, efficient coding model for systems with limited RAM",
        Tags:            []string{"coding", "fast", "efficient"},
    },
    // Medium models (8-16GB RAM)
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
```

## Model Selection Algorithm

### Auto-Selection Logic

```go
func SelectModel(availableRAM int64) (*ModelInfo, error) {
    // Filter by RAM requirement (leave 2GB buffer for OS)
    usableRAM := availableRAM - (2 * 1024 * 1024 * 1024)
    
    candidates := FilterByRAM(ModelRegistry, usableRAM)
    if len(candidates) == 0 {
        return nil, ErrInsufficientRAM
    }
    
    // Prefer recommended models
    recommended := FilterByTag(candidates, "recommended")
    if len(recommended) > 0 {
        candidates = recommended
    }
    
    // Select largest model that fits
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].RAMRequiredMB > candidates[j].RAMRequiredMB
    })
    
    return &candidates[0], nil
}
```

### Selection Tiers

| Available RAM | Recommended Model     | Rationale                          |
|---------------|----------------------|-------------------------------------|
| < 6 GB        | (Not supported)      | Insufficient for any coding model   |
| 6-10 GB       | Qwen 2.5 Coder 3B Q4 | Fast, functional on limited RAM     |
| 10-16 GB      | Qwen 2.5 Coder 7B Q5 | Best balance of speed and quality   |
| 16-24 GB      | Qwen 2.5 Coder 14B Q5| High quality, acceptable speed      |
| 24+ GB        | Qwen 2.5 Coder 14B Q8| Maximum quality (future)            |

## Model Downloading

### Download Sources
1. **Primary**: HuggingFace Hub via HTTPS
2. **Fallback**: (Future) Mirror servers, torrent

### Download Process

```
1. Check if model exists in cache (~/.vibrant/models/)
   ├─ Exists → Verify checksum
   │   ├─ Valid → Use cached model
   │   └─ Invalid → Re-download
   └─ Not exists → Continue to download

2. Create temporary download location
   ~/.vibrant/models/.downloading/<model-id>.gguf.part

3. Download with progress tracking
   ├─ Resume support (HTTP Range headers)
   ├─ Progress bar in terminal
   └─ Speed and ETA display

4. Verify SHA256 checksum
   ├─ Valid → Move to final location
   └─ Invalid → Delete and error

5. Move to final location
   ~/.vibrant/models/<model-id>.gguf

6. Update local manifest
   ~/.vibrant/models/manifest.json
```

### Download Implementation

```go
type Downloader struct {
    CacheDir     string
    Client       *http.Client
    ProgressFunc func(downloaded, total int64)
}

func (d *Downloader) Download(model *ModelInfo) error {
    url := BuildHuggingFaceURL(model)
    destPath := filepath.Join(d.CacheDir, model.ID + ".gguf")
    tempPath := destPath + ".part"
    
    // Check if already downloaded
    if FileExists(destPath) && VerifyChecksum(destPath, model.SHA256) {
        return nil
    }
    
    // Resume support
    var offset int64
    if FileExists(tempPath) {
        offset = FileSize(tempPath)
    }
    
    // Download with progress
    req := NewRequest("GET", url)
    if offset > 0 {
        req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
    }
    
    resp := d.Client.Do(req)
    defer resp.Body.Close()
    
    // Stream to file with progress updates
    return StreamWithProgress(resp.Body, tempPath, offset, d.ProgressFunc)
}
```

## Model Caching

### Cache Structure

```
~/.vibrant/
└── models/
    ├── manifest.json              # Metadata about cached models
    ├── qwen2.5-coder-3b-q4.gguf
    ├── qwen2.5-coder-7b-q5.gguf
    └── .downloading/              # Temporary download location
        └── (temp files)
```

### Manifest Format

```json
{
  "version": "1.0",
  "models": [
    {
      "id": "qwen2.5-coder-3b-q4",
      "path": "qwen2.5-coder-3b-q4.gguf",
      "size_bytes": 1990000000,
      "checksum": "sha256:abc123...",
      "downloaded_at": "2026-01-30T21:00:00Z",
      "last_used": "2026-01-30T21:30:00Z",
      "use_count": 15
    }
  ]
}
```

### Cache Management

```go
type CacheManager struct {
    CacheDir     string
    MaxCacheSizeGB int
    Manifest     *CacheManifest
}

// Operations
func (c *CacheManager) List() []CachedModel
func (c *CacheManager) Get(modelID string) (*CachedModel, error)
func (c *CacheManager) Remove(modelID string) error
func (c *CacheManager) Clear() error
func (c *CacheManager) Prune() error // Remove least-used to stay under limit
func (c *CacheManager) UpdateLastUsed(modelID string)
```

### Cache Policies
- **Max Size**: Configurable (default: unlimited)
- **Pruning**: LRU (Least Recently Used) when size limit reached
- **Auto-cleanup**: Remove failed downloads older than 24h

## Model Loading

### Load Process

```go
type ModelManager struct {
    cache      *CacheManager
    selector   *ModelSelector
    loader     *LLamaLoader
    currentModel *LoadedModel
    mu         sync.Mutex
}

func (m *ModelManager) LoadModel(modelID string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    // Unload existing model
    if m.currentModel != nil {
        m.currentModel.Unload()
    }
    
    // Get model from cache
    cached := m.cache.Get(modelID)
    if cached == nil {
        return ErrModelNotFound
    }
    
    // Load into memory
    model := m.loader.Load(cached.Path, LoadOptions{
        ContextSize: 4096,
        Threads:     runtime.NumCPU(),
        UseMMap:     true,
        UseMlock:    false,
    })
    
    m.currentModel = model
    m.cache.UpdateLastUsed(modelID)
    
    return nil
}
```

### Load Options

```go
type LoadOptions struct {
    ContextSize int    // Max context tokens
    Threads     int    // CPU threads for inference
    BatchSize   int    // Batch size for prompt processing
    UseMMap     bool   // Use memory-mapped files
    UseMlock    bool   // Lock pages in RAM (requires privileges)
    GPU         int    // GPU layers (0 for CPU-only)
}
```

## Model Updates

### Update Strategy
- **Manual**: User runs `vibrant model update`
- **Check Frequency**: Weekly (background check)
- **Notification**: Inform user of available updates
- **Auto-update**: Optional flag (default: off)

### Version Tracking
```go
type ModelVersion struct {
    ModelID      string
    Version      string    // e.g., "v1.5"
    ReleasedAt   time.Time
    Changelog    string
    DownloadURL  string
    Deprecated   bool
}
```

## Error Handling

### Error Types
```go
var (
    ErrInsufficientRAM    = errors.New("insufficient RAM for any model")
    ErrModelNotFound      = errors.New("model not found in registry")
    ErrDownloadFailed     = errors.New("failed to download model")
    ErrChecksumMismatch   = errors.New("checksum verification failed")
    ErrModelLoadFailed    = errors.New("failed to load model")
    ErrCacheFull          = errors.New("cache size limit reached")
)
```

### Recovery Strategies
1. **Download failure**: Retry with exponential backoff (3 attempts)
2. **Checksum failure**: Delete and re-download
3. **Load failure**: Try smaller model or report error
4. **Cache full**: Auto-prune or prompt user

## CLI Commands

```bash
# List available models
vibrant model list [--all | --cached]

# Show model info
vibrant model info <model-id>

# Download a specific model
vibrant model download <model-id>

# Remove a cached model
vibrant model remove <model-id>

# Show cache statistics
vibrant model cache [--usage]

# Clean cache
vibrant model cache clean [--older-than 30d]
```

## Performance Considerations

1. **Parallel Downloads**: Single model at a time (large files)
2. **Checksums**: Computed during download (streaming)
3. **Memory Mapping**: Preferred for model loading (reduces RAM usage)
4. **Lazy Loading**: Don't load model until first inference

## Status

- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Dependencies**: system package (RAM detection)
- **Implementation**: Phase 2 of PLAN.md
