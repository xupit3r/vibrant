// +build !darwin

package gpu

import "fmt"

// MetalDevice stub for non-Darwin platforms
type MetalDevice struct{}

func NewMetalDevice() (*MetalDevice, error) {
	return nil, fmt.Errorf("Metal is only supported on macOS")
}

func (d *MetalDevice) Type() DeviceType { return DeviceTypeGPU }
func (d *MetalDevice) Name() string     { return "Metal (unsupported)" }

func (d *MetalDevice) Allocate(size int64) (Buffer, error) {
	return nil, fmt.Errorf("Metal not supported on this platform")
}

func (d *MetalDevice) Copy(dst, src Buffer, size int64) error {
	return fmt.Errorf("Metal not supported on this platform")
}

func (d *MetalDevice) Sync() error {
	return fmt.Errorf("Metal not supported on this platform")
}

func (d *MetalDevice) Free() error {
	return nil
}

func (d *MetalDevice) MemoryUsage() (int64, int64) {
	return 0, 0
}
