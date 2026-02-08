// +build linux,cgo

package tensor

import (
	"fmt"
	"unsafe"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/cuda"
)

// RoPEGPU applies rotary position embeddings on GPU
// input: [batch_size, num_heads, seq_len, head_dim]
// cosTable, sinTable: precomputed cos/sin values [maxSeqLen * halfDim]
// positions: position indices [seq_len]
// Returns nil if GPU execution fails
func RoPEGPU(input *Tensor, cosTable, sinTable []float32, positions []int) *Tensor {
	return ropeGPU(input, cosTable, sinTable, positions)
}

// ropeGPU is the internal implementation
func ropeGPU(input *Tensor, cosTable, sinTable []float32, positions []int) *Tensor {
	if !input.IsOnGPU() || input.dtype != Float32 {
		return nil
	}

	shape := input.Shape()
	if len(shape) != 4 {
		return nil
	}

	batchSize := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]
	halfDim := headDim / 2

	// Allocate output buffer on GPU
	totalSize := batchSize * numHeads * seqLen * headDim
	outputSize := int64(totalSize * 4) // float32
	outputBuf, err := input.gpuDevice.Allocate(outputSize)
	if err != nil {
		fmt.Printf("CUDA RoPE: failed to allocate output buffer: %v\n", err)
		return nil
	}

	// Allocate and copy cosTable to GPU
	cosSize := int64(len(cosTable) * 4)
	cosBuf, err := input.gpuDevice.Allocate(cosSize)
	if err != nil {
		fmt.Printf("CUDA RoPE: failed to allocate cos buffer: %v\n", err)
		outputBuf.Free()
		return nil
	}
	cosBytes := unsafe.Slice((*byte)(unsafe.Pointer(&cosTable[0])), cosSize)
	if err := cosBuf.CopyFromHost(cosBytes); err != nil {
		fmt.Printf("CUDA RoPE: failed to copy cos table: %v\n", err)
		outputBuf.Free()
		cosBuf.Free()
		return nil
	}

	// Allocate and copy sinTable to GPU
	sinSize := int64(len(sinTable) * 4)
	sinBuf, err := input.gpuDevice.Allocate(sinSize)
	if err != nil {
		fmt.Printf("CUDA RoPE: failed to allocate sin buffer: %v\n", err)
		outputBuf.Free()
		cosBuf.Free()
		return nil
	}
	sinBytes := unsafe.Slice((*byte)(unsafe.Pointer(&sinTable[0])), sinSize)
	if err := sinBuf.CopyFromHost(sinBytes); err != nil {
		fmt.Printf("CUDA RoPE: failed to copy sin table: %v\n", err)
		outputBuf.Free()
		cosBuf.Free()
		sinBuf.Free()
		return nil
	}

	// Convert positions to int32 and copy to GPU
	positions32 := make([]int32, len(positions))
	for i, p := range positions {
		positions32[i] = int32(p)
	}
	posSize := int64(len(positions32) * 4)
	posBuf, err := input.gpuDevice.Allocate(posSize)
	if err != nil {
		fmt.Printf("CUDA RoPE: failed to allocate positions buffer: %v\n", err)
		outputBuf.Free()
		cosBuf.Free()
		sinBuf.Free()
		return nil
	}
	posBytes := unsafe.Slice((*byte)(unsafe.Pointer(&positions32[0])), posSize)
	if err := posBuf.CopyFromHost(posBytes); err != nil {
		fmt.Printf("CUDA RoPE: failed to copy positions: %v\n", err)
		outputBuf.Free()
		cosBuf.Free()
		sinBuf.Free()
		posBuf.Free()
		return nil
	}

	// Get CUDA device and kernels
	cudaDev, ok := input.gpuDevice.(*gpu.CUDADevice)
	if !ok {
		outputBuf.Free()
		cosBuf.Free()
		sinBuf.Free()
		posBuf.Free()
		return nil
	}

	kernels, ok := input.gpuKernels.(*cuda.KernelSet)
	if !ok {
		outputBuf.Free()
		cosBuf.Free()
		sinBuf.Free()
		posBuf.Free()
		return nil
	}

	// Launch kernel
	err = kernels.LaunchRoPE(
		unsafe.Pointer(input.gpuBuffer.Ptr()),
		unsafe.Pointer(outputBuf.Ptr()),
		unsafe.Pointer(cosBuf.Ptr()),
		unsafe.Pointer(sinBuf.Ptr()),
		unsafe.Pointer(posBuf.Ptr()),
		batchSize, numHeads, seqLen, headDim, halfDim,
		nil, // default stream
	)

	if err != nil {
		fmt.Printf("CUDA RoPE: kernel launch failed: %v\n", err)
		cosBuf.Free()
		sinBuf.Free()
		posBuf.Free()
		outputBuf.Free()
		return nil
	}

	// Sync BEFORE freeing temp buffers - kernel may still be reading them
	if err := cudaDev.Sync(); err != nil {
		fmt.Printf("CUDA RoPE: sync failed: %v\n", err)
		cosBuf.Free()
		sinBuf.Free()
		posBuf.Free()
		outputBuf.Free()
		return nil
	}

	// Free temporary buffers after sync (kernel is done reading them)
	cosBuf.Free()
	sinBuf.Free()
	posBuf.Free()

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
