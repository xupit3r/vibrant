package tensor

import (
"testing"
)

func TestWeightCacheStats(t *testing.T) {
// Test that cache stats API works
stats := DefaultWeightCache.DetailedStats()

t.Logf("Initial cache stats: hits=%d, misses=%d, hitRate=%.2f%%, entries=%d, used=%d MB, evictions=%d",
stats.Hits, stats.Misses, stats.HitRate, stats.Entries, stats.UsedBytes/(1024*1024), stats.Evictions)

// Reset stats
DefaultWeightCache.ResetStats()

stats = DefaultWeightCache.DetailedStats()
if stats.Hits != 0 || stats.Misses != 0 || stats.Evictions != 0 {
t.Errorf("Reset didn't work: hits=%d, misses=%d, evictions=%d", stats.Hits, stats.Misses, stats.Evictions)
}

// Verify Stats() still works
used, budget, entries := DefaultWeightCache.Stats()
t.Logf("Cache capacity: used=%d MB, budget=%d GB, entries=%d", 
used/(1024*1024), budget/(1024*1024*1024), entries)

if budget != 32*1024*1024*1024 {
t.Errorf("Expected 32GB budget, got %d bytes", budget)
}
}

func TestCacheEviction(t *testing.T) {
// Create a very small cache to force evictions
cache := &WeightCacheManager{
budget:  4 * 1024, // 4KB - very small
used:    0,
entries: make(map[*Tensor]int64),
}

// Create test tensors (each will be ~256 bytes as float32)
tensors := make([]*Tensor, 20)
for i := 0; i < 20; i++ {
data := make([]float32, 64) // 64 * 4 = 256 bytes
tensors[i] = &Tensor{
shape: []int{8, 8},
dtype: Float32,
data:  data,
}
}

// Register all tensors - should trigger evictions
for _, tensor := range tensors {
cached := &Tensor{
shape: []int{8, 8},
dtype: Float32,
data:  make([]float32, 64),
}
cache.Register(tensor, cached)
}

// Check that cache stayed within budget
used, budget, entries := cache.Stats()
if used > budget {
t.Errorf("Cache exceeded budget: %d > %d", used, budget)
}

t.Logf("Final cache state: used=%d, budget=%d, entries=%d", used, budget, entries)

stats := cache.DetailedStats()
t.Logf("Evictions triggered during test: %d", stats.Evictions)

if stats.Evictions == 0 {
t.Error("Expected some evictions with small cache, got 0")
}
}
