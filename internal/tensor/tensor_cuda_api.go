// +build linux,cgo

package tensor

import "fmt"

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
