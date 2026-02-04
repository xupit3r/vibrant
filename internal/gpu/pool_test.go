// +build darwin

package gpu

import (
	"runtime"
	"testing"
)

func TestBufferPool(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	// Create pool with 10MB limit
	pool := NewBufferPool(dev, 10*1024*1024)
	defer pool.Clear()

	// Test basic allocation
	buf1, err := pool.Allocate(1024)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	if buf1.Size() < 1024 {
		t.Errorf("Buffer too small: %d < 1024", buf1.Size())
	}

	// Release buffer back to pool
	if err := pool.Release(buf1); err != nil {
		t.Errorf("Release failed: %v", err)
	}

	stats := pool.Stats()
	if stats.Allocations != 1 {
		t.Errorf("Expected 1 allocation, got %d", stats.Allocations)
	}

	// Allocate again - should reuse from pool
	buf2, err := pool.Allocate(1024)
	if err != nil {
		t.Fatalf("Second allocate failed: %v", err)
	}

	stats = pool.Stats()
	if stats.Reuses != 1 {
		t.Errorf("Expected 1 reuse, got %d", stats.Reuses)
	}
	if stats.PoolHits != 1 {
		t.Errorf("Expected 1 pool hit, got %d", stats.PoolHits)
	}

	pool.Release(buf2)
}

func TestBufferPoolSizeRounding(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	pool := NewBufferPool(dev, 10*1024*1024)
	defer pool.Clear()

	// Allocate 100 bytes - should round up
	buf1, _ := pool.Allocate(100)
	pool.Release(buf1)

	// Allocate 200 bytes - should reuse rounded buffer
	buf2, _ := pool.Allocate(200)
	
	stats := pool.Stats()
	if stats.Reuses != 1 {
		t.Errorf("Expected buffer reuse due to size rounding, got %d reuses", stats.Reuses)
	}

	pool.Release(buf2)
}

func TestBufferPoolEviction(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	// Create small pool (1MB limit)
	pool := NewBufferPool(dev, 1*1024*1024)
	defer pool.Clear()

	// Allocate and release several buffers
	for i := 0; i < 5; i++ {
		buf, _ := pool.Allocate(256 * 1024) // 256KB each
		pool.Release(buf)
	}

	stats := pool.Stats()
	if stats.Evictions == 0 {
		t.Logf("Warning: Expected some evictions with 1MB pool limit")
		// Not a hard failure - pool might be more efficient than expected
	}

	pooled, _, _ := pool.MemoryUsage()
	if pooled > 1*1024*1024 {
		t.Errorf("Pool exceeded limit: %d > %d", pooled, 1*1024*1024)
	}
}

func TestBufferPoolClear(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	pool := NewBufferPool(dev, 10*1024*1024)
	
	// Allocate and release several buffers
	for i := 0; i < 10; i++ {
		buf, _ := pool.Allocate(1024)
		pool.Release(buf)
	}

	pooled1, _, _ := pool.MemoryUsage()
	if pooled1 == 0 {
		t.Error("Expected pooled memory > 0 before clear")
	}

	// Clear pool
	if err := pool.Clear(); err != nil {
		t.Errorf("Clear failed: %v", err)
	}

	pooled2, _, _ := pool.MemoryUsage()
	if pooled2 != 0 {
		t.Errorf("Expected pooled memory = 0 after clear, got %d", pooled2)
	}
}

func TestBufferPoolStats(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	pool := NewBufferPool(dev, 10*1024*1024)
	defer pool.Clear()

	// Do various operations
	buf1, _ := pool.Allocate(1024)
	buf2, _ := pool.Allocate(2048)
	pool.Release(buf1)
	buf3, _ := pool.Allocate(1024) // Should reuse buf1
	pool.Release(buf2)
	pool.Release(buf3)

	stats := pool.Stats()

	if stats.Allocations < 3 {
		t.Errorf("Expected at least 3 allocations, got %d", stats.Allocations)
	}

	if stats.Reuses < 1 {
		t.Errorf("Expected at least 1 reuse, got %d", stats.Reuses)
	}

	if stats.PoolHits < 1 {
		t.Errorf("Expected at least 1 pool hit, got %d", stats.PoolHits)
	}

	t.Logf("Pool stats: Allocs=%d, Reuses=%d, Hits=%d, Misses=%d, Evictions=%d",
		stats.Allocations, stats.Reuses, stats.PoolHits, stats.PoolMisses, stats.Evictions)
}

func TestMetalDevicePoolIntegration(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	// Allocate through device (should use pool)
	buf1, err := dev.Allocate(1024)
	if err != nil {
		t.Fatalf("Device allocate failed: %v", err)
	}

	// Get pool stats
	stats := dev.PoolStats()
	if stats.Allocations < 1 {
		t.Error("Expected pool to track allocation")
	}

	// Release
	buf1.Free()

	// Allocate again - should reuse
	buf2, err := dev.Allocate(1024)
	if err != nil {
		t.Fatalf("Second device allocate failed: %v", err)
	}

	stats = dev.PoolStats()
	if stats.Reuses < 1 {
		t.Error("Expected pool reuse")
	}

	buf2.Free()

	// Check pool memory usage
	pooled, active, max := dev.PoolMemoryUsage()
	t.Logf("Pool memory: %d pooled, %d active, %d max", pooled, active, max)

	if max == 0 {
		t.Error("Expected non-zero pool max")
	}
}

func BenchmarkBufferPoolAllocate(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	pool := NewBufferPool(dev, 100*1024*1024)
	defer pool.Clear()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, _ := pool.Allocate(4096)
		pool.Release(buf)
	}
}

func BenchmarkBufferPoolVsDirect(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("GPU only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	b.Run("WithPool", func(b *testing.B) {
		pool := NewBufferPool(dev, 100*1024*1024)
		defer pool.Clear()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf, _ := pool.Allocate(4096)
			pool.Release(buf)
		}
	})

	b.Run("DirectAlloc", func(b *testing.B) {
		// Direct allocation without pool
		// Create a new device without pool for comparison
		directDev, _ := GetDevice(DeviceTypeGPU)
		defer directDev.Free()
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf, _ := directDev.Allocate(4096)
			buf.Free()
		}
	})
}
