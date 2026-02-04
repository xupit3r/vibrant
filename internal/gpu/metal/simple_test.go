// +build darwin,cgo

package metal

import (
	"runtime"
	"testing"

	"github.com/xupit3r/vibrant/internal/gpu"
)

// TestSimpleCopy tests the simplest possible kernel (copy)
func TestSimpleCopy(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal only supported on macOS")
	}

	dev, err := gpu.NewMetalDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Free()

	devicePtr := getDevicePtr(dev)
	queuePtr := getQueuePtr(dev)

	ks, err := NewKernelSet(devicePtr)
	if err != nil {
		t.Fatalf("NewKernelSet failed: %v", err)
	}
	defer ks.Free()

	// Test simple copy
	data := []float32{1.0, 2.0, 3.0, 4.0}
	size := uint32(len(data))

	// Allocate buffers
	bufIn, _ := dev.Allocate(int64(len(data) * 4))
	defer bufIn.Free()
	bufOut, _ := dev.Allocate(int64(len(data) * 4))
	defer bufOut.Free()

	// Write input
	bufIn.CopyFromHost(float32SliceToBytes(data))

	t.Logf("Input buffer: %p", bufIn.MetalBuffer())
	t.Logf("Output buffer: %p", bufOut.MetalBuffer())

	// Dispatch copy kernel
	err = ks.DispatchCopy(queuePtr, ElementwiseParams{
		A:    bufIn.MetalBuffer(),
		B:    bufOut.MetalBuffer(),
		Size: size,
	})
	if err != nil {
		t.Fatalf("DispatchCopy failed: %v", err)
	}

	// Sync
	if err := dev.Sync(); err != nil {
		t.Fatalf("Sync failed: %v", err)
	}

	// Read result
	resultBytes := make([]byte, len(data)*4)
	bufOut.CopyToHost(resultBytes)
	result := bytesToFloat32Slice(resultBytes)

	t.Logf("Input: %v", data)
	t.Logf("Output: %v", result)

	// Verify
	for i := range data {
		if result[i] != data[i] {
			t.Errorf("Copy mismatch at index %d: expected %.2f, got %.2f", i, data[i], result[i])
		}
	}
}
