// +build !linux !cgo

package cuda

import (
	"fmt"
	"unsafe"
)

// Stub implementations for non-Linux or non-CGO builds

func LaunchMatMulKernel(A, B, C unsafe.Pointer, M, N, K int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchMatMulSingleRowKernel(A, B, C unsafe.Pointer, N, K int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchSoftmaxKernel(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchSoftmaxBatchedKernel(input, output unsafe.Pointer, batchSize, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchRMSNormKernel(input, weight, output unsafe.Pointer, size int, eps float32, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchRMSNormBatchedKernel(input, weight, output unsafe.Pointer, batchSize, size int, eps float32, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchAddKernel(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchMulKernel(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchMulScalarKernel(A unsafe.Pointer, scalar float32, B unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchSiLUKernel(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchCopyKernel(src, dst unsafe.Pointer, size int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}

func LaunchRoPEKernel(input, output unsafe.Pointer, cosTable, sinTable unsafe.Pointer, positions unsafe.Pointer,
	batchSize, numHeads, seqLen, headDim, halfDim int, stream unsafe.Pointer) error {
	return fmt.Errorf("CUDA is only supported on Linux with CGO enabled")
}
