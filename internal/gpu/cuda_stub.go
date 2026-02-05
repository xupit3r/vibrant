// +build !linux !cgo

package gpu

import "fmt"

// CUDADevice stub for non-Linux or non-CGO builds
type CUDADevice struct{}

// NewCUDADevice returns an error on unsupported platforms
func NewCUDADevice() (*CUDADevice, error) {
	return nil, fmt.Errorf("CUDA support requires Linux with CGO enabled (build with: go build -tags cuda)")
}

func (d *CUDADevice) Type() DeviceType        { return DeviceTypeGPU }
func (d *CUDADevice) Name() string             { return "CUDA (unavailable)" }
func (d *CUDADevice) Allocate(size int64) (Buffer, error) { return nil, fmt.Errorf("CUDA not available") }
func (d *CUDADevice) Copy(dst, src Buffer, size int64) error { return fmt.Errorf("CUDA not available") }
func (d *CUDADevice) Sync() error             { return fmt.Errorf("CUDA not available") }
func (d *CUDADevice) Free() error             { return fmt.Errorf("CUDA not available") }
func (d *CUDADevice) MemoryUsage() (int64, int64) { return 0, 0 }
func (d *CUDADevice) PoolStats() PoolStats    { return PoolStats{} }
func (d *CUDADevice) PoolMemoryUsage() (int64, int64, int64) { return 0, 0, 0 }
