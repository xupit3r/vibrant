// +build !linux !cgo

package cuda

import (
	"fmt"
	"unsafe"
)

// KernelSet stub for non-CUDA builds
type KernelSet struct{}

// Kernel stub
type Kernel struct{}

// NewKernelSet returns an error on unsupported platforms
func NewKernelSet() (*KernelSet, error) {
	return nil, fmt.Errorf("CUDA kernels require Linux with CGO enabled")
}

func (ks *KernelSet) LaunchMatMul(A, B, C unsafe.Pointer, M, N, K int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchMatMulSingleRow(A, B, C unsafe.Pointer, N, K int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchSoftmax(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchSoftmaxBatched(input, output unsafe.Pointer, batchSize, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchRMSNorm(input, weight, output unsafe.Pointer, size int, eps float32, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchRMSNormBatched(input, weight, output unsafe.Pointer, batchSize, size int, eps float32, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchAdd(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchMul(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchMulScalar(A unsafe.Pointer, scalar float32, B unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchSiLU(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) LaunchCopy(src, dst unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available")
}

func (ks *KernelSet) Free() error {
	return nil
}
