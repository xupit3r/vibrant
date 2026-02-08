// +build !linux !cgo
// +build !darwin

package tensor

// GPU operation stubs for non-GPU builds
// These should never be called since ToDevice will fail for GPU

func matmulGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

func softmaxGPU(input *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

func rmsNormGPU(input, weight *Tensor, eps float32) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

func addGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

func mulGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

func siluGPU(input *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS or Linux")
}

// RMSNormGPU stub for non-GPU builds
func RMSNormGPU(input, weight *Tensor, eps float32) *Tensor {
	return nil
}

// RoPEGPU stub for non-GPU builds
func RoPEGPU(input *Tensor, cosTable, sinTable []float32, positions []int) *Tensor {
	return nil
}

// SoftmaxGPU stub for non-GPU builds
func SoftmaxGPU(input *Tensor) *Tensor {
	return nil
}
