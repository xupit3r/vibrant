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

	// Moving to GPU: dequantize if needed
	if targetDevice == GPU {
		// If tensor is quantized, dequantize to Float32 first
		if t.dtype != Float32 {
			fmt.Printf("Dequantizing %s tensor to Float32 for GPU transfer...\n", t.dtype)
			dequantized, err := t.dequantizeForGPU()
			if err != nil {
				return nil, fmt.Errorf("failed to dequantize for GPU: %w", err)
			}
			// Move the dequantized tensor to GPU
			return dequantized.toGPU()
		}
		return t.toGPU()
	}

	// Moving to CPU
	if targetDevice == CPU {
		return t.toCPU()
	}

	return nil, fmt.Errorf("unsupported device: %v", targetDevice)
}

// dequantizeForGPU dequantizes a quantized tensor to Float32
func (t *Tensor) dequantizeForGPU() (*Tensor, error) {
	switch t.dtype {
	case Q4_K:
		return DequantizeQ4_KTensor(t)
	case Q5_K:
		return DequantizeQ5_KTensor(t)
	case Q6_K:
		return DequantizeQ6_KTensor(t)
	case Float32:
		// Already Float32, return as-is
		return t, nil
	default:
		return nil, fmt.Errorf("unsupported quantization format for GPU: %v", t.dtype)
	}
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
