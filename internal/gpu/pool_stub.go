// +build !linux !cgo
// +build !darwin !cgo

package gpu

import "unsafe"

// BufferPool stub for non-Darwin platforms
type BufferPool struct {
	device Device
}

type PoolStats struct {
	Allocations int64
	Reuses      int64
	Evictions   int64
	PoolHits    int64
	PoolMisses  int64
}

type pooledBuffer struct {
	Buffer
}

func NewBufferPool(device Device, maxBytes int64) *BufferPool {
	return &BufferPool{device: device}
}

func (p *BufferPool) Allocate(size int64) (Buffer, error) {
	return p.device.Allocate(size)
}

func (p *BufferPool) Release(buf Buffer) error {
	return buf.Free()
}

func (p *BufferPool) Clear() error {
	return nil
}

func (p *BufferPool) Stats() PoolStats {
	return PoolStats{}
}

func (p *BufferPool) MemoryUsage() (pooled, active, max int64) {
	return 0, 0, 0
}

func (b *pooledBuffer) MetalBuffer() unsafe.Pointer {
	return nil
}
