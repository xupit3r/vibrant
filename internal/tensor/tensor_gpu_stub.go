// +build !linux !cgo
// +build !darwin !cgo

package tensor

import "fmt"

// ToDevice stub for non-GPU builds
func (t *Tensor) ToDevice(targetDevice Device) (*Tensor, error) {
	if targetDevice == GPU {
		return nil, fmt.Errorf("GPU support not available (requires macOS or Linux with CGO)")
	}
	return t, nil
}

// IsOnGPU stub for non-GPU builds
func (t *Tensor) IsOnGPU() bool {
	return false
}

// FreeGPU stub for non-GPU builds
func (t *Tensor) FreeGPU() error {
	return nil
}

// SyncGPU stub for non-GPU builds
func (t *Tensor) SyncGPU() error {
	return nil
}
