package tensor

import (
	"fmt"
	"unsafe"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/metal"
)

// Global kernel set cache (one per device)
var kernelSetCache = make(map[gpu.Device]*metal.KernelSet)

// ToDevice moves the tensor to the specified device
// If the tensor is already on the target device, returns the tensor unchanged
func (t *Tensor) ToDevice(targetDevice Device) (*Tensor, error) {
	// Already on target device
	if t.device == targetDevice {
		return t, nil
	}

	// Only support Float32 on GPU for now
	if targetDevice == GPU && t.dtype != Float32 {
		return nil, fmt.Errorf("GPU only supports Float32 tensors (got %v)", t.dtype)
	}

	// Moving to GPU
	if targetDevice == GPU {
		return t.toGPU()
	}

	// Moving to CPU
	if targetDevice == CPU {
		return t.toCPU()
	}

	return nil, fmt.Errorf("unsupported device: %v", targetDevice)
}

// toGPU moves the tensor to GPU
func (t *Tensor) toGPU() (*Tensor, error) {
	// Get or create GPU device
	gpuDev, err := gpu.GetDevice(gpu.DeviceTypeGPU)
	if err != nil {
		return nil, fmt.Errorf("failed to get GPU device: %w", err)
	}

	// Get or create kernel set for this device
	kernels, ok := kernelSetCache[gpuDev]
	if !ok {
		// Get device pointer from MetalDevice
		metalDev, ok := gpuDev.(*gpu.MetalDevice)
		if !ok {
			return nil, fmt.Errorf("GPU device is not a MetalDevice")
		}

		// Extract device pointer (this is a bit hacky but necessary)
		devicePtr := getMetalDevicePtr(metalDev)
		kernels, err = metal.NewKernelSet(devicePtr)
		if err != nil {
			return nil, fmt.Errorf("failed to create kernel set: %w", err)
		}
		kernelSetCache[gpuDev] = kernels
	}

	// Calculate buffer size
	var dataSlice []float32
	switch d := t.data.(type) {
	case []float32:
		dataSlice = d
	default:
		return nil, fmt.Errorf("cannot move %T to GPU (only []float32 supported)", t.data)
	}

	bufferSize := int64(len(dataSlice) * 4) // 4 bytes per float32

	// Allocate GPU buffer
	buf, err := gpuDev.Allocate(bufferSize)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate GPU buffer: %w", err)
	}

	// Copy data to GPU
	dataBytes := (*[1 << 30]byte)(unsafe.Pointer(&dataSlice[0]))[:bufferSize:bufferSize]
	if err := buf.CopyFromHost(dataBytes); err != nil {
		buf.Free()
		return nil, fmt.Errorf("failed to copy data to GPU: %w", err)
	}

	// Create new tensor on GPU
	gpuTensor := &Tensor{
		data:       dataSlice, // Keep CPU copy for fallback
		shape:      append([]int{}, t.shape...),
		stride:     append([]int{}, t.stride...),
		dtype:      t.dtype,
		device:     GPU,
		gpuBuffer:  buf,
		gpuDevice:  gpuDev,
		gpuKernels: kernels,
		transposed: t.transposed,
	}

	return gpuTensor, nil
}

// toCPU moves the tensor from GPU to CPU
func (t *Tensor) toCPU() (*Tensor, error) {
	if t.device != GPU {
		return t, nil
	}

	if t.gpuBuffer == nil {
		return nil, fmt.Errorf("GPU tensor has no buffer")
	}

	// Allocate CPU memory
	size := t.gpuBuffer.Size()
	cpuData := make([]float32, size/4)

	// Copy from GPU
	dataBytes := (*[1 << 30]byte)(unsafe.Pointer(&cpuData[0]))[:size:size]
	if err := t.gpuBuffer.CopyToHost(dataBytes); err != nil {
		return nil, fmt.Errorf("failed to copy from GPU: %w", err)
	}

	// Create CPU tensor
	cpuTensor := &Tensor{
		data:       cpuData,
		shape:      append([]int{}, t.shape...),
		stride:     append([]int{}, t.stride...),
		dtype:      t.dtype,
		device:     CPU,
		transposed: t.transposed,
	}

	return cpuTensor, nil
}

// IsOnGPU returns true if the tensor is on GPU
func (t *Tensor) IsOnGPU() bool {
	return t.device == GPU && t.gpuBuffer != nil
}

// FreeGPU releases GPU resources
func (t *Tensor) FreeGPU() error {
	if t.gpuBuffer != nil {
		err := t.gpuBuffer.Free()
		t.gpuBuffer = nil
		return err
	}
	return nil
}

// SyncGPU waits for all GPU operations on this tensor to complete
func (t *Tensor) SyncGPU() error {
	if t.gpuDevice != nil {
		return t.gpuDevice.Sync()
	}
	return nil
}

// Helper to extract Metal device pointer
// This uses unsafe to access the internal structure
func getMetalDevicePtr(metalDev *gpu.MetalDevice) unsafe.Pointer {
	// The MetalDevice struct has ctx as first field, which is *MetalContext
	// MetalContext has device as first field
	type metalContext struct {
		device       unsafe.Pointer
		commandQueue unsafe.Pointer
	}
	// Get ctx field (first field of MetalDevice)
	devPtr := (*[2]unsafe.Pointer)(unsafe.Pointer(metalDev))
	ctx := (*metalContext)(devPtr[0])
	return ctx.device
}

// Helper to get command queue pointer
func getMetalQueuePtr(metalDev *gpu.MetalDevice) unsafe.Pointer {
	type metalContext struct {
		device       unsafe.Pointer
		commandQueue unsafe.Pointer
	}
	devPtr := (*[2]unsafe.Pointer)(unsafe.Pointer(metalDev))
	ctx := (*metalContext)(devPtr[0])
	return ctx.commandQueue
}
