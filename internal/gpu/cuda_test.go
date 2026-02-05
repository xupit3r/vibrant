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
