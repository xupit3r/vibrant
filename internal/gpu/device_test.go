package gpu

import (
	"runtime"
	"testing"
)

func TestGetDefaultDevice(t *testing.T) {
	dev, err := GetDefaultDevice()
	if err != nil {
		t.Fatalf("GetDefaultDevice failed: %v", err)
	}
	defer dev.Free()

	if dev == nil {
		t.Fatal("GetDefaultDevice returned nil device")
	}

	name := dev.Name()
	if name == "" {
		t.Error("Device name is empty")
	}
	t.Logf("Default device: %s (type: %v)", name, dev.Type())
}

func TestGetCPUDevice(t *testing.T) {
	dev, err := GetDevice(DeviceTypeCPU)
	if err != nil {
		t.Fatalf("GetDevice(CPU) failed: %v", err)
	}
	defer dev.Free()

	if dev.Type() != DeviceTypeCPU {
		t.Errorf("Expected CPU device, got %v", dev.Type())
	}

	t.Logf("CPU device: %s", dev.Name())
}

func TestGetGPUDevice(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := GetDevice(DeviceTypeGPU)
	if err != nil {
		t.Skipf("GPU device not available: %v", err)
	}
	defer dev.Free()

	if dev.Type() != DeviceTypeGPU {
		t.Errorf("Expected GPU device, got %v", dev.Type())
	}

	t.Logf("GPU device: %s", dev.Name())

	// Check memory info
	used, total := dev.MemoryUsage()
	t.Logf("GPU memory: %d MB used / %d MB total", used/(1024*1024), total/(1024*1024))
}

func TestCPUBufferAllocate(t *testing.T) {
	dev := NewCPUDevice()
	defer dev.Free()

	sizes := []int64{1024, 1024 * 1024, 16 * 1024 * 1024}

	for _, size := range sizes {
		buf, err := dev.Allocate(size)
		if err != nil {
			t.Fatalf("Allocate(%d) failed: %v", size, err)
		}
		defer buf.Free()

		if buf.Size() != size {
			t.Errorf("Buffer size mismatch: expected %d, got %d", size, buf.Size())
		}
	}
}

func TestCPUBufferCopy(t *testing.T) {
	dev := NewCPUDevice()
	defer dev.Free()

	size := int64(1024)
	buf1, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	defer buf1.Free()

	buf2, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	defer buf2.Free()

	// Write test data to buf1
	testData := make([]byte, size)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	if err := buf1.CopyFromHost(testData); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	// Copy buf1 to buf2
	if err := dev.Copy(buf2, buf1, size); err != nil {
		t.Fatalf("Copy failed: %v", err)
	}

	// Read back from buf2
	result := make([]byte, size)
	if err := buf2.CopyToHost(result); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	// Verify data
	for i := range result {
		if result[i] != testData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, testData[i], result[i])
		}
	}
}

func TestGPUBufferAllocate(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	sizes := []int64{1024, 1024 * 1024, 16 * 1024 * 1024}

	for _, size := range sizes {
		buf, err := dev.Allocate(size)
		if err != nil {
			t.Fatalf("Allocate(%d) failed: %v", size, err)
		}
		defer buf.Free()

		if buf.Size() != size {
			t.Errorf("Buffer size mismatch: expected %d, got %d", size, buf.Size())
		}

		if buf.Ptr() == 0 {
			t.Error("Buffer pointer is null")
		}
	}

	// Check memory usage increased
	used, total := dev.MemoryUsage()
	t.Logf("GPU memory after allocations: %d MB used / %d MB total", 
		used/(1024*1024), total/(1024*1024))
}

func TestGPUBufferHostTransfer(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	size := int64(4096)
	buf, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	defer buf.Free()

	// Create test data
	testData := make([]byte, size)
	for i := range testData {
		testData[i] = byte((i * 7) % 256)
	}

	// Copy to GPU
	if err := buf.CopyFromHost(testData); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	// Copy back from GPU
	result := make([]byte, size)
	if err := buf.CopyToHost(result); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	// Verify data
	for i := range result {
		if result[i] != testData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, testData[i], result[i])
		}
	}
}

func TestGPUBufferCopy(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	size := int64(8192)
	buf1, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate buf1 failed: %v", err)
	}
	defer buf1.Free()

	buf2, err := dev.Allocate(size)
	if err != nil {
		t.Fatalf("Allocate buf2 failed: %v", err)
	}
	defer buf2.Free()

	// Write test data to buf1
	testData := make([]byte, size)
	for i := range testData {
		testData[i] = byte((i * 13) % 256)
	}
	if err := buf1.CopyFromHost(testData); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	// Copy buf1 to buf2 on GPU
	if err := dev.Copy(buf2, buf1, size); err != nil {
		t.Fatalf("GPU Copy failed: %v", err)
	}

	// Sync to ensure copy is complete
	if err := dev.Sync(); err != nil {
		t.Fatalf("Sync failed: %v", err)
	}

	// Read back from buf2
	result := make([]byte, size)
	if err := buf2.CopyToHost(result); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	// Verify data
	for i := range result {
		if result[i] != testData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, testData[i], result[i])
			break // Only report first mismatch
		}
	}
}

func TestGPUSync(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	// Sync should not fail even with no pending operations
	if err := dev.Sync(); err != nil {
		t.Errorf("Sync failed: %v", err)
	}
}

func TestBufferFree(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	buf, err := dev.Allocate(1024)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	// Free should succeed
	if err := buf.Free(); err != nil {
		t.Errorf("Free failed: %v", err)
	}

	// Double-free should be safe (no-op)
	if err := buf.Free(); err != nil {
		t.Errorf("Second Free failed: %v", err)
	}
}

func TestDeviceTypeString(t *testing.T) {
	tests := []struct {
		dt   DeviceType
		want string
	}{
		{DeviceTypeCPU, "CPU"},
		{DeviceTypeGPU, "GPU"},
		{DeviceType(999), "Unknown"},
	}

	for _, tt := range tests {
		got := tt.dt.String()
		if got != tt.want {
			t.Errorf("DeviceType(%d).String() = %s, want %s", tt.dt, got, tt.want)
		}
	}
}

// Benchmark GPU buffer allocation
func BenchmarkGPUAllocate(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	size := int64(1024 * 1024) // 1MB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, err := dev.Allocate(size)
		if err != nil {
			b.Fatalf("Allocate failed: %v", err)
		}
		buf.Free()
	}
}

// Benchmark GPU host-to-device transfer
func BenchmarkGPUHostToDevice(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	size := int64(4 * 1024 * 1024) // 4MB
	buf, err := dev.Allocate(size)
	if err != nil {
		b.Fatalf("Allocate failed: %v", err)
	}
	defer buf.Free()

	data := make([]byte, size)

	b.SetBytes(size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyFromHost(data); err != nil {
			b.Fatalf("CopyFromHost failed: %v", err)
		}
	}
}

// Benchmark GPU device-to-host transfer
func BenchmarkGPUDeviceToHost(b *testing.B) {
	if runtime.GOOS != "darwin" {
		b.Skip("GPU device only supported on macOS")
	}

	dev, err := NewMetalDevice()
	if err != nil {
		b.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	size := int64(4 * 1024 * 1024) // 4MB
	buf, err := dev.Allocate(size)
	if err != nil {
		b.Fatalf("Allocate failed: %v", err)
	}
	defer buf.Free()

	data := make([]byte, size)

	b.SetBytes(size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.CopyToHost(data); err != nil {
			b.Fatalf("CopyToHost failed: %v", err)
		}
	}
}
