// +build linux,cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart

#include <cuda_runtime.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// KernelSet contains all compiled CUDA kernels
type KernelSet struct {
	// Matrix operations
	MatMul          *Kernel
	MatMulSingleRow *Kernel

	// Normalization
	Softmax        *Kernel
	SoftmaxBatched *Kernel
	RMSNorm        *Kernel
	RMSNormBatched *Kernel

	// Element-wise operations
	Add       *Kernel
	Mul       *Kernel
	MulScalar *Kernel
	SiLU      *Kernel
	Copy      *Kernel

	// Kernel function pointers (stored as void*)
	kernelFuncs map[string]unsafe.Pointer
}

// Kernel represents a compiled CUDA kernel function
type Kernel struct {
	name     string
	funcPtr  unsafe.Pointer
}

// NewKernelSet loads all CUDA kernels
// Note: This assumes kernels are linked at compile time
func NewKernelSet() (*KernelSet, error) {
	ks := &KernelSet{
		kernelFuncs: make(map[string]unsafe.Pointer),
	}

	// In a linked binary, we can get kernel function pointers
	// For now, we'll create placeholder kernels
	// Actual kernel loading will be done when we compile the .cu file

	kernelNames := []string{
		"matmul_f32",
		"matmul_f32_single_row",
		"softmax_f32",
		"softmax_batched_f32",
		"rms_norm_f32",
		"rms_norm_batched_f32",
		"add_f32",
		"mul_f32",
		"mul_scalar_f32",
		"silu_f32",
		"copy_f32",
	}

	for _, name := range kernelNames {
		kernel := &Kernel{
			name:    name,
			funcPtr: nil, // Will be populated when kernels are loaded
		}

		switch name {
		case "matmul_f32":
			ks.MatMul = kernel
		case "matmul_f32_single_row":
			ks.MatMulSingleRow = kernel
		case "softmax_f32":
			ks.Softmax = kernel
		case "softmax_batched_f32":
			ks.SoftmaxBatched = kernel
		case "rms_norm_f32":
			ks.RMSNorm = kernel
		case "rms_norm_batched_f32":
			ks.RMSNormBatched = kernel
		case "add_f32":
			ks.Add = kernel
		case "mul_f32":
			ks.Mul = kernel
		case "mul_scalar_f32":
			ks.MulScalar = kernel
		case "silu_f32":
			ks.SiLU = kernel
		case "copy_f32":
			ks.Copy = kernel
		}
	}

	return ks, nil
}

// LaunchMatMul launches the matrix multiplication kernel
// A: [M x K], B: [K x N], C: [M x N]
func (ks *KernelSet) LaunchMatMul(
	A, B, C unsafe.Pointer,
	M, N, K int,
	stream unsafe.Pointer,
) error {
	if ks.MatMul == nil {
		return fmt.Errorf("matmul kernel not loaded")
	}
	return LaunchMatMulKernel(A, B, C, M, N, K, stream)
}

// LaunchMatMulSingleRow launches single-row matmul (for decode phase)
// A: [1 x K], B: [K x N], C: [1 x N]
func (ks *KernelSet) LaunchMatMulSingleRow(
	A, B, C unsafe.Pointer,
	N, K int,
	stream unsafe.Pointer,
) error {
	if ks.MatMulSingleRow == nil {
		return fmt.Errorf("matmul_single_row kernel not loaded")
	}
	return LaunchMatMulSingleRowKernel(A, B, C, N, K, stream)
}

// LaunchSoftmax launches the softmax kernel
func (ks *KernelSet) LaunchSoftmax(
	input, output unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.Softmax == nil {
		return fmt.Errorf("softmax kernel not loaded")
	}
	return LaunchSoftmaxKernel(input, output, size, stream)
}

// LaunchSoftmaxBatched launches the batched softmax kernel
func (ks *KernelSet) LaunchSoftmaxBatched(
	input, output unsafe.Pointer,
	batchSize, size int,
	stream unsafe.Pointer,
) error {
	if ks.SoftmaxBatched == nil {
		return fmt.Errorf("softmax_batched kernel not loaded")
	}
	return LaunchSoftmaxBatchedKernel(input, output, batchSize, size, stream)
}

// LaunchRMSNorm launches the RMS normalization kernel
func (ks *KernelSet) LaunchRMSNorm(
	input, weight, output unsafe.Pointer,
	size int,
	eps float32,
	stream unsafe.Pointer,
) error {
	if ks.RMSNorm == nil {
		return fmt.Errorf("rms_norm kernel not loaded")
	}
	return LaunchRMSNormKernel(input, weight, output, size, eps, stream)
}

// LaunchRMSNormBatched launches the batched RMS normalization kernel
func (ks *KernelSet) LaunchRMSNormBatched(
	input, weight, output unsafe.Pointer,
	batchSize, size int,
	eps float32,
	stream unsafe.Pointer,
) error {
	if ks.RMSNormBatched == nil {
		return fmt.Errorf("rms_norm_batched kernel not loaded")
	}
	return LaunchRMSNormBatchedKernel(input, weight, output, batchSize, size, eps, stream)
}

// LaunchAdd launches the element-wise addition kernel
func (ks *KernelSet) LaunchAdd(
	A, B, C unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.Add == nil {
		return fmt.Errorf("add kernel not loaded")
	}
	return LaunchAddKernel(A, B, C, size, stream)
}

// LaunchMul launches the element-wise multiplication kernel
func (ks *KernelSet) LaunchMul(
	A, B, C unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.Mul == nil {
		return fmt.Errorf("mul kernel not loaded")
	}
	return LaunchMulKernel(A, B, C, size, stream)
}

// LaunchMulScalar launches the scalar multiplication kernel
func (ks *KernelSet) LaunchMulScalar(
	A unsafe.Pointer,
	scalar float32,
	B unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.MulScalar == nil {
		return fmt.Errorf("mul_scalar kernel not loaded")
	}
	return LaunchMulScalarKernel(A, scalar, B, size, stream)
}

// LaunchSiLU launches the SiLU activation kernel
func (ks *KernelSet) LaunchSiLU(
	input, output unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.SiLU == nil {
		return fmt.Errorf("silu kernel not loaded")
	}
	return LaunchSiLUKernel(input, output, size, stream)
}

// LaunchCopy launches the copy kernel
func (ks *KernelSet) LaunchCopy(
	src, dst unsafe.Pointer,
	size int,
	stream unsafe.Pointer,
) error {
	if ks.Copy == nil {
		return fmt.Errorf("copy kernel not loaded")
	}
	return LaunchCopyKernel(src, dst, size, stream)
}

// Free releases kernel resources
func (ks *KernelSet) Free() error {
	// Cleanup will be added when we have actual kernel management
	ks.kernelFuncs = nil
	return nil
}
