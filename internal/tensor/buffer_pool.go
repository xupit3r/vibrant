package tensor

import (
	"sync"
)

// TensorBufferPool manages a pool of reusable tensor buffers
// This reduces GC pressure by reusing memory for temporary tensors
type TensorBufferPool struct {
	pools map[int]*sync.Pool // size -> pool of []float32
	mu    sync.RWMutex
}

// GlobalTensorPool is the global tensor buffer pool
var GlobalTensorPool = &TensorBufferPool{
	pools: make(map[int]*sync.Pool),
}

// Get retrieves a buffer of the specified size from the pool
// If no buffer is available, allocates a new one
func (p *TensorBufferPool) Get(size int) []float32 {
	if size <= 0 {
		return nil
	}

	p.mu.RLock()
	pool, exists := p.pools[size]
	p.mu.RUnlock()

	if !exists {
		// Create new pool for this size
		p.mu.Lock()
		// Check again after acquiring write lock
		pool, exists = p.pools[size]
		if !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return make([]float32, size)
				},
			}
			p.pools[size] = pool
		}
		p.mu.Unlock()
	}

	return pool.Get().([]float32)
}

// Put returns a buffer to the pool for reuse
// The buffer will be zeroed before being returned to the pool
func (p *TensorBufferPool) Put(buf []float32) {
	if buf == nil || len(buf) == 0 {
		return
	}

	size := len(buf)

	p.mu.RLock()
	pool, exists := p.pools[size]
	p.mu.RUnlock()

	if !exists {
		// Pool doesn't exist for this size, just let GC handle it
		return
	}

	// Zero the buffer before returning to pool
	// This prevents data leakage between uses
	for i := range buf {
		buf[i] = 0
	}

	pool.Put(buf)
}

// GetTensor allocates a new tensor using a pooled buffer
// The tensor should be returned to the pool via PutTensor when done
func (p *TensorBufferPool) GetTensor(shape []int, dtype DataType) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	var data interface{}
	switch dtype {
	case Float32:
		data = p.Get(size)
	default:
		// For non-float32 types, allocate normally
		return NewTensor(shape, dtype)
	}

	return &Tensor{
		shape:  shape,
		dtype:  dtype,
		data:   data,
		device: CPU,
		pooled: true, // Mark as pooled so we can return it later
	}
}

// PutTensor returns a pooled tensor's buffer back to the pool
// Only works for tensors allocated via GetTensor
func (p *TensorBufferPool) PutTensor(t *Tensor) {
	if t == nil || !t.pooled {
		return
	}

	if t.dtype == Float32 {
		if buf, ok := t.data.([]float32); ok {
			p.Put(buf)
			t.data = nil // Prevent double-free
		}
	}
}

// Clear empties all pools (useful for testing)
func (p *TensorBufferPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools = make(map[int]*sync.Pool)
}
