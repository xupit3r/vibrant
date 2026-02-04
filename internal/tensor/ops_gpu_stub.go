// +build !darwin !cgo

package tensor

// GPU operation stubs for non-GPU builds
// These should never be called since ToDevice will fail for GPU

func matmulGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS")
}

func softmaxGPU(input *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS")
}

func rmsNormGPU(input, weight *Tensor, eps float32) *Tensor {
	panic("GPU operations not available without CGO on macOS")
}

func addGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS")
}

func mulGPU(a, b *Tensor) *Tensor {
	panic("GPU operations not available without CGO on macOS")
}
