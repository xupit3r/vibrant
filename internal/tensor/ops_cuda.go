// +build linux,cgo

package tensor

import (
	"fmt"
	"unsafe"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/cuda"
)

// matmulGPU performs matrix multiplication on GPU using CUDA
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
		fmt.Printf("CUDA MatMul: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernel set
	cudaDev, ok := a.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := a.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Launch kernel (stream is nil for default stream)
	var launchErr error
	if M == 1 {
		// Single-row optimization for decode step
		launchErr = kernels.LaunchMatMulSingleRow(
			unsafe.Pointer(a.gpuBuffer.Ptr()),
			unsafe.Pointer(b.gpuBuffer.Ptr()),
			unsafe.Pointer(outputBuf.Ptr()),
			N, K,
			nil, // default stream
		)
	} else {
		// General matrix multiplication
		launchErr = kernels.LaunchMatMul(
			unsafe.Pointer(a.gpuBuffer.Ptr()),
			unsafe.Pointer(b.gpuBuffer.Ptr()),
			unsafe.Pointer(outputBuf.Ptr()),
			M, N, K,
			nil, // default stream
		)
	}

	if launchErr != nil {
		outputBuf.Free()
		return nil
	}

	// Sync to ensure completion
	if err := cudaDev.Sync(); err != nil {
		outputBuf.Free()
		return nil
	}

	fmt.Printf("[DEBUG] CUDA MatMul: sync completed\n")

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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

// softmaxGPU performs softmax on GPU using CUDA
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
	outputSize := int64(size * 4) // float32
	outputBuf, err := input.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA Softmax: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := input.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := input.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Launch kernel
	err = kernels.LaunchSoftmax(
		unsafe.Pointer(input.gpuBuffer.Ptr()),
		unsafe.Pointer(outputBuf.Ptr()),
		size,
		nil, // default stream
	)

	if err != nil {
		fmt.Printf("CUDA Softmax: kernel launch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA Softmax: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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

// rmsNormGPU performs RMS normalization on GPU using CUDA
// Input can be [batch, seq, hidden] or [batch*seq, hidden] — normalizes along last dimension
// Weight must be [hidden_dim] matching the last dimension of input
func rmsNormGPU(input, weight *Tensor, eps float32) *Tensor {
	if !input.IsOnGPU() || !weight.IsOnGPU() {
		return nil
	}

	if input.dtype != Float32 || weight.dtype != Float32 {
		return nil
	}

	totalSize := 1
	for _, dim := range input.shape {
		totalSize *= dim
	}

	// The normalization dimension is the last dimension (must match weight size)
	normDim := input.shape[len(input.shape)-1]
	batchSize := totalSize / normDim

	// Allocate output buffer
	outputSize := int64(totalSize * 4)
	outputBuf, err := input.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA RMSNorm: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := input.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := input.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Use batched kernel: each "batch" is one row to normalize
	if batchSize == 1 {
		// Single row — use simple kernel
		err = kernels.LaunchRMSNorm(
			unsafe.Pointer(input.gpuBuffer.Ptr()),
			unsafe.Pointer(weight.gpuBuffer.Ptr()),
			unsafe.Pointer(outputBuf.Ptr()),
			normDim,
			eps,
			nil, // default stream
		)
	} else {
		// Multiple rows — use batched kernel
		err = kernels.LaunchRMSNormBatched(
			unsafe.Pointer(input.gpuBuffer.Ptr()),
			unsafe.Pointer(weight.gpuBuffer.Ptr()),
			unsafe.Pointer(outputBuf.Ptr()),
			batchSize,
			normDim,
			eps,
			nil, // default stream
		)
	}

	if err != nil {
		fmt.Printf("CUDA RMSNorm: kernel launch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA RMSNorm: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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

// addGPU performs element-wise addition on GPU using CUDA
func addGPU(a, b *Tensor) *Tensor {
	if !a.IsOnGPU() || !b.IsOnGPU() {
		return nil
	}

	if a.dtype != Float32 || b.dtype != Float32 {
		return nil
	}

	size := 1
	for _, dim := range a.shape {
		size *= dim
	}

	// Allocate output buffer
	outputSize := int64(size * 4)
	outputBuf, err := a.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA Add: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := a.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := a.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Launch kernel
	err = kernels.LaunchAdd(
		unsafe.Pointer(a.gpuBuffer.Ptr()),
		unsafe.Pointer(b.gpuBuffer.Ptr()),
		unsafe.Pointer(outputBuf.Ptr()),
		size,
		nil, // default stream
	)

	if err != nil {
		fmt.Printf("CUDA Add: kernel launch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA Add: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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

// mulGPU performs element-wise multiplication on GPU using CUDA
func mulGPU(a, b *Tensor) *Tensor {
	if !a.IsOnGPU() || !b.IsOnGPU() {
		return nil
	}

	if a.dtype != Float32 || b.dtype != Float32 {
		return nil
	}

	size := 1
	for _, dim := range a.shape {
		size *= dim
	}

	// Allocate output buffer
	outputSize := int64(size * 4)
	outputBuf, err := a.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA Mul: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := a.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := a.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Launch kernel
	err = kernels.LaunchMul(
		unsafe.Pointer(a.gpuBuffer.Ptr()),
		unsafe.Pointer(b.gpuBuffer.Ptr()),
		unsafe.Pointer(outputBuf.Ptr()),
		size,
		nil, // default stream
	)

	if err != nil {
		fmt.Printf("CUDA Mul: kernel launch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA Mul: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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

// RMSNormGPU is an exported wrapper for rmsNormGPU
func RMSNormGPU(input, weight *Tensor, eps float32) *Tensor {
	return rmsNormGPU(input, weight, eps)
}

// SoftmaxGPU is an exported wrapper for softmaxGPU
func SoftmaxGPU(input *Tensor) *Tensor {
	return softmaxGPU(input)
}

// siluGPU performs SiLU activation on GPU using CUDA
func siluGPU(input *Tensor) *Tensor {
	if !input.IsOnGPU() || input.dtype != Float32 {
		return nil
	}

	size := 1
	for _, dim := range input.shape {
		size *= dim
	}

	// Allocate output buffer
	outputSize := int64(size * 4)
	outputBuf, err := input.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA SiLU: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := input.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		return nil
	}

	kernels, ok := input.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		return nil
	}

	// Launch kernel
	err = kernels.LaunchSiLU(
		unsafe.Pointer(input.gpuBuffer.Ptr()),
		unsafe.Pointer(outputBuf.Ptr()),
		size,
		nil, // default stream
	)

	if err != nil {
		fmt.Printf("CUDA SiLU: kernel launch failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Sync
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA SiLU: sync failed: %v\n", err)
		outputBuf.Free()
		return nil
	}

	// Create output tensor (data=nil so EnsureCPUData will transfer from GPU)
	outputTensor := &Tensor{
		data:       nil,
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
