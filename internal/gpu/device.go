package gpu

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Device represents a compute device (CPU or GPU)
type Device interface {
	// Type returns the device type
	Type() DeviceType

	// Name returns a human-readable device name
	Name() string

	// Allocate allocates a buffer of the given size in bytes
	Allocate(size int64) (Buffer, error)

	// Copy copies data from src to dst buffer
	Copy(dst, src Buffer, size int64) error

	// Sync waits for all pending operations to complete
	Sync() error

	// Free releases the device and all associated resources
	Free() error

	// MemoryUsage returns current GPU memory usage in bytes (used, total)
	MemoryUsage() (int64, int64)
}

// DeviceType represents the type of compute device
type DeviceType int

const (
	DeviceTypeCPU DeviceType = iota
	DeviceTypeGPU
)

func (dt DeviceType) String() string {
	switch dt {
	case DeviceTypeCPU:
		return "CPU"
	case DeviceTypeGPU:
		return "GPU"
	default:
		return "Unknown"
	}
}

// GetDefaultDevice returns the default device for the current system
// On Apple Silicon, returns Metal GPU device if available, otherwise CPU
func GetDefaultDevice() (Device, error) {
	// Check if we're on macOS with Metal support
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm64" || runtime.GOARCH == "amd64") {
		dev, err := NewMetalDevice()
		if err == nil {
			return dev, nil
		}
		// Fall back to CPU if Metal initialization fails
	}

	return NewCPUDevice(), nil
}

// GetDevice returns a device of the specified type
func GetDevice(dtype DeviceType) (Device, error) {
	switch dtype {
	case DeviceTypeCPU:
		return NewCPUDevice(), nil
	case DeviceTypeGPU:
		// Try Metal on macOS
		if runtime.GOOS == "darwin" {
			return NewMetalDevice()
		}
		// Try CUDA on Linux
		if runtime.GOOS == "linux" {
			return NewCUDADevice()
		}
		return nil, fmt.Errorf("GPU not supported on %s", runtime.GOOS)
	default:
		return nil, fmt.Errorf("unknown device type: %v", dtype)
	}
}

// CPUDevice represents the CPU device (no-op for compatibility)
type CPUDevice struct {
	name string
}

// NewCPUDevice creates a new CPU device
func NewCPUDevice() *CPUDevice {
	return &CPUDevice{
		name: fmt.Sprintf("CPU (%s)", runtime.GOARCH),
	}
}

func (d *CPUDevice) Type() DeviceType { return DeviceTypeCPU }
func (d *CPUDevice) Name() string     { return d.name }

func (d *CPUDevice) Allocate(size int64) (Buffer, error) {
	// CPU "buffers" are just regular Go slices
	return &cpuBuffer{data: make([]byte, size)}, nil
}

func (d *CPUDevice) Copy(dst, src Buffer, size int64) error {
	dstBuf, ok := dst.(*cpuBuffer)
	if !ok {
		return fmt.Errorf("dst is not a CPU buffer")
	}
	srcBuf, ok := src.(*cpuBuffer)
	if !ok {
		return fmt.Errorf("src is not a CPU buffer")
	}
	copy(dstBuf.data[:size], srcBuf.data[:size])
	return nil
}

func (d *CPUDevice) Sync() error {
	// No-op for CPU
	return nil
}

func (d *CPUDevice) Free() error {
	// No-op for CPU
	return nil
}

func (d *CPUDevice) MemoryUsage() (int64, int64) {
	// Return 0 for both - CPU memory tracking is handled elsewhere
	return 0, 0
}

// cpuBuffer implements Buffer for CPU memory
type cpuBuffer struct {
	data []byte
	mu   sync.RWMutex
}

func (b *cpuBuffer) Size() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return int64(len(b.data))
}

func (b *cpuBuffer) Ptr() uintptr {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if len(b.data) == 0 {
		return 0
	}
	return uintptr(0) // Not meaningful for CPU buffers
}

func (b *cpuBuffer) CopyToHost(dst []byte) error {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if int64(len(dst)) < int64(len(b.data)) {
		return fmt.Errorf("destination buffer too small: %d < %d", len(dst), len(b.data))
	}
	copy(dst, b.data)
	return nil
}

func (b *cpuBuffer) CopyFromHost(src []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if int64(len(b.data)) < int64(len(src)) {
		return fmt.Errorf("buffer too small: %d < %d", len(b.data), len(src))
	}
	copy(b.data, src)
	return nil
}

func (b *cpuBuffer) Free() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = nil
	return nil
}

func (b *cpuBuffer) Device() Device {
	return NewCPUDevice()
}

func (b *cpuBuffer) MetalBuffer() unsafe.Pointer {
	return nil
}
