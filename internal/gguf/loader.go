package gguf

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/tensor"
)

// LoadTensor loads a tensor from the GGUF file using memory mapping
// This is the preferred method as it avoids copying large amounts of data
func (g *GGUFFile) LoadTensor(name string) (*tensor.Tensor, error) {
	info, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found in GGUF file", name)
	}

	// Calculate absolute offset in file
	absoluteOffset := g.tensorDataOffset + int64(info.Offset)

	// Convert GGML type to tensor DataType
	dtype := ggmlTypeToTensorType(info.Type)

	// Convert dimensions to int slice
	shape := make([]int, len(info.Dims))
	for i, dim := range info.Dims {
		shape[i] = int(dim)
	}

	// Calculate expected size based on shape and dtype
	expectedElements := 1
	for _, dim := range shape {
		expectedElements *= dim
	}
	var expectedSize int64
	switch dtype {
	case tensor.Float32:
		expectedSize = int64(expectedElements * 4)
	case tensor.Q5_K:
		// Q5_K uses 176 bytes per 256 elements
		numBlocks := (expectedElements + 255) / 256
		expectedSize = int64(numBlocks * 176)
	case tensor.Q6_K:
		// Q6_K uses 210 bytes per 256 elements
		numBlocks := (expectedElements + 255) / 256
		expectedSize = int64(numBlocks * 210)
	default:
		expectedSize = int64(expectedElements * dtype.BytesPerElement())
	}

	// Fix shape mismatch for Float32 tensors (common issue in some GGUF files)
	if int64(info.Size) != expectedSize && dtype == tensor.Float32 {
		actualElements := int(info.Size) / 4 // float32 is 4 bytes
		if len(shape) == 2 && actualElements%shape[0] == 0 {
			// Fix the second dimension based on actual data size
			correctedShape := []int{shape[0], actualElements / shape[0]}
			fmt.Printf("Info: Correcting shape for %s from %v to %v (actual size=%d bytes)\n",
				name, shape, correctedShape, info.Size)
			shape = correctedShape
			expectedSize = int64(info.Size)
		} else {
			fmt.Printf("WARNING: Tensor %s size mismatch - metadata says %d bytes, expected %d bytes for shape %v dtype %s\n",
				name, info.Size, expectedSize, shape, dtype)
		}
	}

	// Memory-map the tensor data
	t, err := tensor.NewTensorMmap(g.path, absoluteOffset, int64(info.Size), shape, dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to mmap tensor %s: %w", name, err)
	}

	return t, nil
}

// LoadTensorEager loads a tensor from the GGUF file
// For now, this is implemented using mmap (same as LoadTensor) since mmap provides
// lazy loading by default. A future enhancement could copy the data into memory.
func (g *GGUFFile) LoadTensorEager(name string) (*tensor.Tensor, error) {
	// For now, just use mmap which is already lazy
	// A true eager implementation would require tensor package support for
	// creating tensors from raw byte slices while preserving dtype
	return g.LoadTensor(name)
}

// GetTensorInfo returns information about a tensor without loading it
func (g *GGUFFile) GetTensorInfo(name string) (*TensorInfo, error) {
	info, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found in GGUF file", name)
	}
	return info, nil
}

// ListTensors returns a list of all tensor names in the file
func (g *GGUFFile) ListTensors() []string {
	names := make([]string, 0, len(g.Tensors))
	for name := range g.Tensors {
		names = append(names, name)
	}
	return names
}
