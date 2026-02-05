// +build darwin linux
// +build cgo

package gpu

import (
	"fmt"
	"sync"
	"unsafe"
)

// BufferPool manages a pool of GPU buffers for efficient reuse
type BufferPool struct {
	device   Device
	pools    map[int64][]*pooledBuffer // Size -> available buffers
	active   map[uintptr]*pooledBuffer  // Ptr -> active buffers
	mu       sync.RWMutex
	maxBytes int64 // Maximum total bytes to pool
	curBytes int64 // Current pooled bytes
	stats    PoolStats
}

// PoolStats tracks buffer pool statistics
type PoolStats struct {
	Allocations int64 // Total allocations
	Reuses      int64 // Buffers reused from pool
	Evictions   int64 // Buffers evicted due to memory pressure
	PoolHits    int64 // Successful pool lookups
	PoolMisses  int64 // Failed pool lookups (allocated new)
}

// pooledBuffer wraps a buffer with reference counting
type pooledBuffer struct {
	Buffer
	requestedSize int64 // Size requested by user
	actualSize    int64 // Actual allocation size
	poolKey       int64 // Key used for pool storage (rounded for reuse)
	refCount      int32
	pool          *BufferPool
	inUse         bool
}

// NewBufferPool creates a new buffer pool
// maxBytes: maximum memory to keep in pool (0 = unlimited)
func NewBufferPool(device Device, maxBytes int64) *BufferPool {
	return &BufferPool{
		device:   device,
		pools:    make(map[int64][]*pooledBuffer),
		active:   make(map[uintptr]*pooledBuffer),
		maxBytes: maxBytes,
	}
}

// Allocate gets a buffer from the pool or allocates a new one
func (p *BufferPool) Allocate(size int64) (Buffer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.stats.Allocations++

	// Look for available buffer in pool that's big enough
	// Check power-of-2 sizes from requested size up
	poolSize := roundUpPowerOf2(size)
	for checkSize := poolSize; checkSize <= poolSize*2; checkSize *= 2 {
		if buffers, ok := p.pools[checkSize]; ok && len(buffers) > 0 {
			// Reuse buffer from pool
			buf := buffers[len(buffers)-1]
			p.pools[checkSize] = buffers[:len(buffers)-1]
			
			buf.inUse = true
			buf.refCount = 1
			buf.requestedSize = size // Update requested size for reused buffer
			buf.poolKey = poolSize    // Update pool key
			p.active[buf.Ptr()] = buf
			
			p.curBytes -= buf.actualSize
			p.stats.Reuses++
			p.stats.PoolHits++
			
			return buf, nil
		}
	}

	p.stats.PoolMisses++

	// Allocate new buffer at exact requested size (no rounding)
	// This keeps the interface clean - buffer.Size() returns what was requested
	rawBuf, err := p.allocateDirect(size)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU buffer: %w", err)
	}

	// Use rounded size as pool key for better reuse (reuse poolSize from above)
	poolBuf := &pooledBuffer{
		Buffer:        rawBuf,
		requestedSize: size,
		actualSize:    size,
		poolKey:       poolSize, // Store with rounded key for reuse
		refCount:      1,
		pool:          p,
		inUse:         true,
	}

	p.active[rawBuf.Ptr()] = poolBuf

	return poolBuf, nil
}

// directAllocator is implemented by devices that support direct (non-pooled) allocation
type directAllocator interface {
	allocateDirect(size int64) (Buffer, error)
}

// allocateDirect calls the device's direct allocation method
func (p *BufferPool) allocateDirect(size int64) (Buffer, error) {
	if da, ok := p.device.(directAllocator); ok {
		return da.allocateDirect(size)
	}
	// Fallback for devices without direct allocation
	return p.device.Allocate(size)
}

// Release returns a buffer to the pool
func (p *BufferPool) Release(buf Buffer) error {
	// Check if this is a pooled buffer
	poolBuf, ok := buf.(*pooledBuffer)
	if !ok {
		// Not a pooled buffer, just free it directly
		return buf.Free()
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	ptr := buf.Ptr()
	tracked, isTracked := p.active[ptr]
	if !isTracked {
		// Not tracked in pool, free directly
		return poolBuf.Buffer.Free()
	}

	// Decrement reference count
	tracked.refCount--
	if tracked.refCount > 0 {
		return nil // Still in use
	}

	// Return to pool
	delete(p.active, ptr)
	tracked.inUse = false

	// Check if we have space in pool
	if p.maxBytes > 0 && p.curBytes+tracked.actualSize > p.maxBytes {
		// Pool is full, evict oldest buffer
		p.evictOldest()
	}

	// Add to pool using poolKey (rounded size) for better reuse
	p.pools[tracked.poolKey] = append(p.pools[tracked.poolKey], tracked)
	p.curBytes += tracked.actualSize

	return nil
}

// evictOldest removes the oldest buffer from the pool
func (p *BufferPool) evictOldest() {
	// Find a non-empty pool
	for size, buffers := range p.pools {
		if len(buffers) > 0 {
			// Remove first buffer (oldest)
			buf := buffers[0]
			p.pools[size] = buffers[1:]
			p.curBytes -= size
			p.stats.Evictions++
			
			// Actually free the buffer
			buf.Buffer.Free()
			return
		}
	}
}

// Clear empties the pool and frees all cached buffers
func (p *BufferPool) Clear() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var firstErr error

	// Free all pooled buffers
	for size, buffers := range p.pools {
		for _, buf := range buffers {
			if err := buf.Buffer.Free(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		delete(p.pools, size)
	}

	p.curBytes = 0

	return firstErr
}

// Stats returns current pool statistics
func (p *BufferPool) Stats() PoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.stats
}

// MemoryUsage returns current pooled memory and total capacity
func (p *BufferPool) MemoryUsage() (pooled, active, max int64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	activeBytes := int64(0)
	for _, buf := range p.active {
		activeBytes += buf.actualSize
	}

	return p.curBytes, activeBytes, p.maxBytes
}

// roundUpPowerOf2 rounds up to the nearest power of 2
func roundUpPowerOf2(n int64) int64 {
	if n <= 0 {
		return 0
	}
	
	// Handle small sizes specially for efficiency
	if n <= 256 {
		return 256
	}
	if n <= 1024 {
		return 1024
	}
	if n <= 4096 {
		return 4096
	}
	
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++
	
	return n
}

// pooledBuffer methods that delegate to underlying buffer

func (b *pooledBuffer) Size() int64 {
	// Return requested size, not actual allocation size
	return b.requestedSize
}

func (b *pooledBuffer) Ptr() uintptr {
	return b.Buffer.Ptr()
}

func (b *pooledBuffer) CopyToHost(dst []byte) error {
	return b.Buffer.CopyToHost(dst)
}

func (b *pooledBuffer) CopyFromHost(src []byte) error {
	return b.Buffer.CopyFromHost(src)
}

func (b *pooledBuffer) Free() error {
	// Route through pool for proper management
	if b.pool != nil {
		return b.pool.Release(b)
	}
	return b.Buffer.Free()
}

func (b *pooledBuffer) Device() Device {
	return b.Buffer.Device()
}

func (b *pooledBuffer) MetalBuffer() unsafe.Pointer {
	return b.Buffer.MetalBuffer()
}
