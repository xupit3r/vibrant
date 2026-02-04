package tensor

import (
	"sync"
	"sync/atomic"
)

// Global generation counter for LRU ordering
var globalCacheGen uint64

// WeightCacheManager manages a memory-budgeted cache of dequantized weight tensors
type WeightCacheManager struct {
	mu        sync.Mutex
	budget    int64             // Maximum bytes allowed for cache
	used      int64             // Current bytes used
	entries   map[*Tensor]int64 // Tensor -> byte size of cached data
	hits      uint64            // Cache hits (atomic)
	misses    uint64            // Cache misses (atomic)
	evictions uint64            // Number of evictions (atomic)
}

// DefaultWeightCache is the global weight cache instance
var DefaultWeightCache = &WeightCacheManager{
	budget:  32 * 1024 * 1024 * 1024, // 32GB default (increased from 8GB to fix cache thrashing)
	used:    0,
	entries: make(map[*Tensor]int64),
}

// SetBudget sets the maximum cache budget in bytes
func (m *WeightCacheManager) SetBudget(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.budget = bytes
}

// Register registers a cached tensor and evicts LRU entries if over budget
func (m *WeightCacheManager) Register(owner *Tensor, cached *Tensor) {
	size := int64(cached.Size()) * 4 // float32 = 4 bytes

	m.mu.Lock()
	defer m.mu.Unlock()

	// Evict LRU entries if over budget
	for m.used+size > m.budget && len(m.entries) > 0 {
		m.evictLRU()
	}

	m.entries[owner] = size
	m.used += size
}

// evictLRU evicts the least recently used cache entry (must be called with lock held)
func (m *WeightCacheManager) evictLRU() {
	var oldestTensor *Tensor
	var oldestGen uint64 = ^uint64(0)

	for t := range m.entries {
		if t.cacheGen < oldestGen {
			oldestGen = t.cacheGen
			oldestTensor = t
		}
	}

	if oldestTensor != nil {
		size := m.entries[oldestTensor]
		delete(m.entries, oldestTensor)
		m.used -= size
		oldestTensor.dequantCache = nil
		atomic.AddUint64(&m.evictions, 1)
	}
}

// Stats returns cache statistics
func (m *WeightCacheManager) Stats() (used, budget int64, entries int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.used, m.budget, len(m.entries)
}

// DetailedStats returns detailed cache statistics including hit/miss rates
func (m *WeightCacheManager) DetailedStats() CacheStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	hits := atomic.LoadUint64(&m.hits)
	misses := atomic.LoadUint64(&m.misses)
	evictions := atomic.LoadUint64(&m.evictions)
	
	total := hits + misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100
	}
	
	return CacheStats{
		UsedBytes:    m.used,
		BudgetBytes:  m.budget,
		Entries:      len(m.entries),
		Hits:         hits,
		Misses:       misses,
		Evictions:    evictions,
		HitRate:      hitRate,
	}
}

// CacheStats holds detailed cache statistics
type CacheStats struct {
	UsedBytes    int64   // Current bytes used
	BudgetBytes  int64   // Maximum bytes allowed
	Entries      int     // Number of cached entries
	Hits         uint64  // Cache hits
	Misses       uint64  // Cache misses
	Evictions    uint64  // Number of evictions
	HitRate      float64 // Hit rate percentage
}

// ResetStats resets hit/miss/eviction counters
func (m *WeightCacheManager) ResetStats() {
	atomic.StoreUint64(&m.hits, 0)
	atomic.StoreUint64(&m.misses, 0)
	atomic.StoreUint64(&m.evictions, 0)
}

// nextCacheGen atomically increments and returns the next cache generation counter
func nextCacheGen() uint64 {
	return atomic.AddUint64(&globalCacheGen, 1)
}
