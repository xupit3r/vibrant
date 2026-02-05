// +build linux,cgo

package tensor

import (
	"fmt"
	"unsafe"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/cuda"
)

// Global CUDA kernel set cache (one per device)
var kernelSetCache = make(map[gpu.Device]*cuda.KernelSet)

// toGPU moves the tensor to CUDA GPU
func (t *Tensor) toGPU() (*Tensor, error) {
	// Get or create CUDA device
	cudaDev, err := gpu.NewCUDADevice()
	if err != nil {
		return nil, fmt.Errorf("failed to get CUDA device: %w", err)
	}

	// Get or create kernel set for this device
	kernels, ok := kernelSetCache[cudaDev]
	if !ok {
		kernels, err = cuda.NewKernelSet()
		if err != nil {
			return nil, fmt.Errorf("failed to create CUDA kernel set: %w", err)
		}
		kernelSetCache[cudaDev] = kernels
	}

	// Calculate buffer size
	var dataSlice []float32
	switch d := t.data.(type) {
	case []float32:
		dataSlice = d
	default:
		return nil, fmt.Errorf("cannot move %T to CUDA GPU (only []float32 supported)", t.data)
	}

	bufferSize := int64(len(dataSlice) * 4) // 4 bytes per float32

	// Allocate GPU buffer
	buf, err := cudaDev.Allocate(bufferSize)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate CUDA buffer: %w", err)
	}

	// Copy data to GPU
	dataBytes := (*[1 << 30]byte)(unsafe.Pointer(&dataSlice[0]))[:bufferSize:bufferSize]
	if err := buf.CopyFromHost(dataBytes); err != nil {
		buf.Free()
		return nil, fmt.Errorf("failed to copy data to CUDA GPU: %w", err)
	}

	// Create new tensor on GPU
	cudaTensor := &Tensor{
		data:       dataSlice, // Keep CPU copy for fallback
		shape:      append([]int{}, t.shape...),
		stride:     append([]int{}, t.stride...),
		dtype:      t.dtype,
		device:     GPU,
		gpuBuffer:  buf,
		gpuDevice:  cudaDev,
		gpuKernels: kernels,
		transposed: t.transposed,
	}

	return cudaTensor, nil
}

// toCPU moves the tensor from CUDA GPU to CPU
func (t *Tensor) toCPU() (*Tensor, error) {
	if t.device != GPU {
		return t, nil
	}

	// Allocate CPU buffer
	var dataSlice []float32
	switch d := t.data.(type) {
	case []float32:
		dataSlice = make([]float32, len(d))
	default:
		return nil, fmt.Errorf("cannot move %T from CUDA GPU to CPU", t.data)
	}

	bufferSize := int64(len(dataSlice) * 4)
	dataBytes := (*[1 << 30]byte)(unsafe.Pointer(&dataSlice[0]))[:bufferSize:bufferSize]

	// Copy data from GPU
	if err := t.gpuBuffer.CopyToHost(dataBytes); err != nil {
		return nil, fmt.Errorf("failed to copy data from CUDA GPU: %w", err)
	}

	// Create CPU tensor
	cpuTensor := &Tensor{
		data:       dataSlice,
		shape:      append([]int{}, t.shape...),
		stride:     append([]int{}, t.stride...),
		dtype:      t.dtype,
		device:     CPU,
		transposed: t.transposed,
	}

	return cpuTensor, nil
}
