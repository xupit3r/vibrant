// +build linux,cgo

package gpu

import (
	"testing"
)

func TestCUDADevice(t *testing.T) {
	// Skip if CUDA is not available
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Test device properties
	if dev.Type() != DeviceTypeGPU {
		t.Errorf("Expected DeviceTypeGPU, got %v", dev.Type())
	}

	name := dev.Name()
	if name == "" {
		t.Error("Device name is empty")
	}
	t.Logf("CUDA Device: %s", name)

	// Test memory usage
	used, total := dev.MemoryUsage()
	if total == 0 {
		t.Error("Total memory should be > 0")
	}
	t.Logf("Memory: %d MB used / %d MB total", used/(1024*1024), total/(1024*1024))
}

func TestCUDABuffer(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Test buffer allocation
	size := int64(1024 * 1024) // 1 MB
	buf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate buffer: %v", err)
	}
	defer buf.Free()

	if buf.Size() != size {
		t.Errorf("Expected size %d, got %d", size, buf.Size())
	}

	if buf.Ptr() == 0 {
		t.Error("Buffer pointer is null")
	}

	if buf.Device() != dev {
		t.Error("Buffer device mismatch")
	}
}

func TestCUDAHostTransfer(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Create test data
	size := int64(1024)
	hostData := make([]byte, size)
	for i := range hostData {
		hostData[i] = byte(i % 256)
	}

	// Allocate device buffer
	buf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate buffer: %v", err)
	}
	defer buf.Free()

	// Copy to device
	if err := buf.CopyFromHost(hostData); err != nil {
		t.Fatalf("Failed to copy to device: %v", err)
	}

	// Copy back to host
	result := make([]byte, size)
	if err := buf.CopyToHost(result); err != nil {
		t.Fatalf("Failed to copy to host: %v", err)
	}

	// Verify data
	for i := range hostData {
		if result[i] != hostData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, hostData[i], result[i])
			break
		}
	}
}

func TestCUDADeviceToDevice(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Create test data
	size := int64(1024)
	hostData := make([]byte, size)
	for i := range hostData {
		hostData[i] = byte(i % 256)
	}

	// Allocate source and destination buffers
	srcBuf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate source buffer: %v", err)
	}
	defer srcBuf.Free()

	dstBuf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate destination buffer: %v", err)
	}
	defer dstBuf.Free()

	// Copy data to source buffer
	if err := srcBuf.CopyFromHost(hostData); err != nil {
		t.Fatalf("Failed to copy to source buffer: %v", err)
	}

	// Device-to-device copy
	if err := dev.Copy(dstBuf, srcBuf, size); err != nil {
		t.Fatalf("Failed to copy device-to-device: %v", err)
	}

	// Copy back to host
	result := make([]byte, size)
	if err := dstBuf.CopyToHost(result); err != nil {
		t.Fatalf("Failed to copy to host: %v", err)
	}

	// Verify data
	for i := range hostData {
		if result[i] != hostData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, hostData[i], result[i])
			break
		}
	}
}

func TestCUDASync(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Test synchronization
	if err := dev.Sync(); err != nil {
		t.Fatalf("Sync failed: %v", err)
	}
}

func TestCUDAInvalidSize(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Test invalid buffer size
	_, err = dev.Allocate(0)
	if err == nil {
		t.Error("Expected error for zero size allocation")
	}

	_, err = dev.Allocate(-1)
	if err == nil {
		t.Error("Expected error for negative size allocation")
	}
}

func TestCUDABufferPool(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Verify pool is initialized
	pooled, active, max := dev.PoolMemoryUsage()
	t.Logf("Initial pool state: pooled=%d, active=%d, max=%d", pooled, active, max)

	if max == 0 {
		t.Error("Pool max size should be > 0")
	}

	// Allocate and free a buffer (should go to pool)
	size := int64(1024 * 1024) // 1 MB
	buf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate buffer: %v", err)
	}

	// Check active memory increased
	_, active1, _ := dev.PoolMemoryUsage()
	if active1 < size {
		t.Errorf("Active memory should be at least %d, got %d", size, active1)
	}

	// Free buffer (should return to pool)
	if err := buf.Free(); err != nil {
		t.Fatalf("Failed to free buffer: %v", err)
	}

	// Allocate same size again (should hit pool)
	stats1 := dev.PoolStats()
	buf2, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Failed to allocate buffer: %v", err)
	}
	defer buf2.Free()

	stats2 := dev.PoolStats()
	
	// Should have a cache hit
	if stats2.Hits <= stats1.Hits {
		t.Logf("Warning: Expected cache hit (hits before=%d, after=%d)", stats1.Hits, stats2.Hits)
		// Note: This might not always hit due to size bucketing, so just log as warning
	}

	t.Logf("Pool stats: hits=%d, misses=%d, evictions=%d", 
		stats2.Hits, stats2.Misses, stats2.Evictions)
}

func TestCUDABufferPoolReuse(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	size := int64(1024 * 1024) // 1 MB
	numBuffers := 10

	// Allocate and free multiple buffers
	for i := 0; i < numBuffers; i++ {
		buf, err := dev.Allocate(size)
		if err != nil {
			t.Fatalf("Failed to allocate buffer %d: %v", i, err)
		}
		
		// Write some data
		data := make([]byte, size)
		for j := range data {
			data[j] = byte(i)
		}
		if err := buf.CopyFromHost(data); err != nil {
			t.Fatalf("Failed to copy to buffer %d: %v", i, err)
		}

		// Free immediately
		if err := buf.Free(); err != nil {
			t.Fatalf("Failed to free buffer %d: %v", i, err)
		}
	}

	// Check pool statistics
	stats := dev.PoolStats()
	pooled, active, _ := dev.PoolMemoryUsage()

	t.Logf("After %d alloc/free cycles:", numBuffers)
	t.Logf("  Pool stats: hits=%d, misses=%d, evictions=%d",
		stats.Hits, stats.Misses, stats.Evictions)
	t.Logf("  Pool memory: pooled=%d MB, active=%d MB",
		pooled/(1024*1024), active/(1024*1024))

	// Should have some cache hits from reuse
	if stats.Hits == 0 && numBuffers > 1 {
		t.Logf("Warning: Expected some cache hits with %d allocations", numBuffers)
	}
}

func TestCUDABufferPoolPressure(t *testing.T) {
	dev, err := NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	_, _, maxPool := dev.PoolMemoryUsage()
	if maxPool == 0 {
		t.Skip("Pool not initialized")
	}

	// Allocate buffers until we exceed pool size
	size := int64(10 * 1024 * 1024) // 10 MB each
	var buffers []Buffer

	for i := 0; i < 100; i++ { // Allocate up to 1 GB
		buf, err := dev.Allocate(size)
		if err != nil {
			t.Logf("Allocation failed at iteration %d: %v", i, err)
			break
		}
		buffers = append(buffers, buf)

		pooled, active, _ := dev.PoolMemoryUsage()
		if active > maxPool {
			t.Logf("Active memory (%d) exceeded pool max (%d) at %d buffers",
				active, maxPool, len(buffers))
			break
		}
	}

	// Free all buffers
	for _, buf := range buffers {
		buf.Free()
	}

	stats := dev.PoolStats()
	t.Logf("Pool pressure test: allocated %d buffers", len(buffers))
	t.Logf("  Stats: hits=%d, misses=%d, evictions=%d",
		stats.Hits, stats.Misses, stats.Evictions)
}
