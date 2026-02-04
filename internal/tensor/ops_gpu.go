// +build darwin,cgo

package tensor

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/metal"
)

// matmulGPU performs matrix multiplication on GPU using Metal
// A: [M x K], B: [K x N] -> C: [M x N]
// Returns nil if GPU execution fails (caller should fallback to CPU)
func matmulGPU(a, b *Tensor) *Tensor {
	if !a.IsOnGPU() || !b.IsOnGPU() {
		return nil
	}

	if a.dtype != Float32 || b.dtype != Float32 {
		return nil
	}

	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]

	if K != K2 {
		panic(fmt.Sprintf("MatMul dimension mismatch: [%d x %d] @ [%d x %d]", M, K, K2, N))
	}

	// Allocate output buffer on GPU
	outputSize := int64(M * N * 4) // float32
	outputBuf, err := a.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("GPU MatMul: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get Metal device for command queue
	metalDev, ok := a.gpuDevice.(*gpu.MetalDevice)
	if !ok {
		outputBuf.Free()
		return nil
	}
	queuePtr := getMetalQueuePtr(metalDev)

	// Dispatch kernel
	// Use single-row optimization for M=1 (decode step)
	singleRow := M == 1

	kernels := a.gpuKernels.(*metal.KernelSet)
	err = kernels.DispatchMatMul(queuePtr, metal.MatMulParams{
		A:         a.gpuBuffer.MetalBuffer(),
		B:         b.gpuBuffer.MetalBuffer(),
		C:         outputBuf.MetalBuffer(),
		M:         uint32(M),
		N:         uint32(N),
		K:         uint32(K),
		SingleRow: singleRow,
	})

	if err != nil {
		fmt.Printf("GPU MatMul: kernel dispatch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync to ensure completion
	if err := a.gpuDevice.Sync(); err != nil {
		fmt.Printf("GPU MatMul: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor
	outputData := make([]float32, M*N) // Keep CPU copy
	outputTensor := &Tensor{
		data:       outputData,
		shape:      []int{M, N},
		stride:     []int{N, 1},
		dtype:      Float32,
		device:     GPU,
		gpuBuffer:  outputBuf,
		gpuDevice:  a.gpuDevice,
		gpuKernels: a.gpuKernels,
	}

	return outputTensor
}

// softmaxGPU performs softmax on GPU
// input: [size]
// Returns nil if GPU execution fails
func softmaxGPU(input *Tensor) *Tensor {
	if !input.IsOnGPU() || input.dtype != Float32 {
		return nil
	}

	size := 1
	for _, dim := range input.shape {
		size *= dim
	}

	// Allocate output buffer
	outputBuf, err := input.gpuDevice.Allocate(int64(size * 4))
	if err != nil {
		return nil
	}

	metalDev, ok := input.gpuDevice.(*gpu.MetalDevice)
	if !ok {
		outputBuf.Free()
		return nil
	}
	queuePtr := getMetalQueuePtr(metalDev)

	// Dispatch kernel
	kernels := input.gpuKernels.(*metal.KernelSet)
	err = kernels.DispatchSoftmax(queuePtr, metal.SoftmaxParams{
		Input:  input.gpuBuffer.MetalBuffer(),
		Output: outputBuf.MetalBuffer(),
		Size:   uint32(size),
	})

	if err != nil {
		outputBuf.Free()
		return nil
	}

	input.gpuDevice.Sync()

	// Create output tensor
	outputData := make([]float32, size)
	outputTensor := &Tensor{
		data:       outputData,
		shape:      append([]int{}, input.shape...),
		stride:     append([]int{}, input.stride...),
		dtype:      Float32,
		device:     GPU,
		gpuBuffer:  outputBuf,
		gpuDevice:  input.gpuDevice,
		gpuKernels: input.gpuKernels,
	}

	return outputTensor
}

// rmsNormGPU performs RMS normalization on GPU
// input: [size], weight: [size]
// Returns nil if GPU execution fails
func rmsNormGPU(input, weight *Tensor, eps float32) *Tensor {
	if !input.IsOnGPU() || !weight.IsOnGPU() {
		return nil
	}

	if input.dtype != Float32 || weight.dtype != Float32 {
		return nil
	}

	size := 1
	for _, dim := range input.shape {
		size *= dim
	}

	// Allocate output buffer
	outputBuf, err := input.gpuDevice.Allocate(int64(size * 4))
	if err != nil {
		return nil
	}

	metalDev, ok := input.gpuDevice.(*gpu.MetalDevice)
	if !ok {
		outputBuf.Free()
		return nil
	}
	queuePtr := getMetalQueuePtr(metalDev)

	// Dispatch kernel
	kernels := input.gpuKernels.(*metal.KernelSet)
	err = kernels.DispatchRMSNorm(queuePtr, metal.RMSNormParams{
		Input:   input.gpuBuffer.MetalBuffer(),
		Weight:  weight.gpuBuffer.MetalBuffer(),
		Output:  outputBuf.MetalBuffer(),
		Size:    uint32(size),
		Epsilon: eps,
	})

	if err != nil {
		outputBuf.Free()
		return nil
	}

	input.gpuDevice.Sync()

	// Create output tensor
	outputData := make([]float32, size)
	outputTensor := &Tensor{
		data:       outputData,
		shape:      append([]int{}, input.shape...),
		stride:     append([]int{}, input.stride...),
		dtype:      Float32,
		device:     GPU,
		gpuBuffer:  outputBuf,
		gpuDevice:  input.gpuDevice,
		gpuKernels: input.gpuKernels,
	}

	return outputTensor
}

// addGPU performs element-wise addition on GPU
// a, b: same shape
// Returns nil if GPU execution fails
func addGPU(a, b *Tensor) *Tensor {
	if !a.IsOnGPU() || !b.IsOnGPU() {
		return nil
	}

	size := 1
	for _, dim := range a.shape {
		size *= dim
	}

	// Allocate output buffer
	outputBuf, err := a.gpuDevice.Allocate(int64(size * 4))
	if err != nil {
		return nil
	}

	metalDev, ok := a.gpuDevice.(*gpu.MetalDevice)
	if !ok {
		outputBuf.Free()
		return nil
	}
	queuePtr := getMetalQueuePtr(metalDev)

	// Dispatch kernel
	kernels := a.gpuKernels.(*metal.KernelSet)
	err = kernels.DispatchAdd(queuePtr, metal.ElementwiseParams{
		A:    a.gpuBuffer.MetalBuffer(),
		B:    b.gpuBuffer.MetalBuffer(),
		C:    outputBuf.MetalBuffer(),
		Size: uint32(size),
	})

	if err != nil {
		outputBuf.Free()
		return nil
	}

	a.gpuDevice.Sync()

	// Create output tensor
	outputData := make([]float32, size)
	outputTensor := &Tensor{
		data:       outputData,
		shape:      append([]int{}, a.shape...),
		stride:     append([]int{}, a.stride...),
		dtype:      Float32,
		device:     GPU,
		gpuBuffer:  outputBuf,
		gpuDevice:  a.gpuDevice,
		gpuKernels: a.gpuKernels,
	}

	return outputTensor
}

// mulGPU performs element-wise multiplication on GPU
func mulGPU(a, b *Tensor) *Tensor {
	if !a.IsOnGPU() || !b.IsOnGPU() {
		return nil
	}

	size := 1
	for _, dim := range a.shape {
		size *= dim
	}

	// Allocate output buffer
	outputBuf, err := a.gpuDevice.Allocate(int64(size * 4))
	if err != nil {
		return nil
	}

	metalDev, ok := a.gpuDevice.(*gpu.MetalDevice)
	if !ok {
		outputBuf.Free()
		return nil
	}
	queuePtr := getMetalQueuePtr(metalDev)

	// Dispatch kernel
	kernels := a.gpuKernels.(*metal.KernelSet)
	err = kernels.DispatchMul(queuePtr, metal.ElementwiseParams{
		A:    a.gpuBuffer.MetalBuffer(),
		B:    b.gpuBuffer.MetalBuffer(),
		C:    outputBuf.MetalBuffer(),
		Size: uint32(size),
	})

	if err != nil {
		outputBuf.Free()
		return nil
	}

	a.gpuDevice.Sync()

	// Create output tensor
	outputData := make([]float32, size)
	outputTensor := &Tensor{
		data:       outputData,
		shape:      append([]int{}, a.shape...),
		stride:     append([]int{}, a.stride...),
		dtype:      Float32,
		device:     GPU,
		gpuBuffer:  outputBuf,
		gpuDevice:  a.gpuDevice,
		gpuKernels: a.gpuKernels,
	}

	return outputTensor
}
