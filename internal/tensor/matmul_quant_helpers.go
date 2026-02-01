package tensor

import "fmt"

// QuantizeQ5_KTensor converts a Float32 tensor to Q5_K format.
// This is a helper function primarily for testing.
func QuantizeQ5_KTensor(t *Tensor) (*Tensor, error) {
	if t.DType() != Float32 {
		return nil, fmt.Errorf("input tensor must be Float32, got %s", t.DType())
	}

	data := t.data.([]float32)

	// Quantize the data
	blocks, err := QuantizeQ5_K(data)
	if err != nil {
		return nil, err
	}

	// Convert blocks to byte array
	blockSize := len(blocks)
	byteData := make([]byte, blockSize*Q5_K_BLOCK_SIZE)

	for i, block := range blocks {
		offset := i * Q5_K_BLOCK_SIZE
		// Copy D (2 bytes)
		byteData[offset] = byte(block.D)
		byteData[offset+1] = byte(block.D >> 8)
		// Copy Dmin (2 bytes)
		byteData[offset+2] = byte(block.Dmin)
		byteData[offset+3] = byte(block.Dmin >> 8)
		// Copy Scales (12 bytes)
		copy(byteData[offset+4:offset+16], block.Scales[:])
		// Copy Qh (32 bytes)
		copy(byteData[offset+16:offset+48], block.Qh[:])
		// Copy Qs (128 bytes)
		copy(byteData[offset+48:offset+176], block.Qs[:])
	}

	// Create output tensor
	// Compute stride (row-major layout)
	stride := make([]int, len(t.shape))
	if len(t.shape) > 0 {
		stride[len(t.shape)-1] = 1
		for i := len(t.shape) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * t.shape[i+1]
		}
	}

	return &Tensor{
		shape:  t.shape,
		dtype:  Q5_K,
		data:   byteData,
		stride: stride,
		device: CPU,
		offset: 0,
	}, nil
}

// QuantizeQ6_KTensor converts a Float32 tensor to Q6_K format.
// This is a helper function primarily for testing.
func QuantizeQ6_KTensor(t *Tensor) (*Tensor, error) {
	if t.DType() != Float32 {
		return nil, fmt.Errorf("input tensor must be Float32, got %s", t.DType())
	}

	data := t.data.([]float32)

	// Quantize the data
	blocks, err := QuantizeQ6_K(data)
	if err != nil {
		return nil, err
	}

	// Convert blocks to byte array
	blockSize := len(blocks)
	byteData := make([]byte, blockSize*Q6_K_BLOCK_SIZE)

	for i, block := range blocks {
		offset := i * Q6_K_BLOCK_SIZE
		// Copy Ql (128 bytes)
		copy(byteData[offset:offset+128], block.Ql[:])
		// Copy Qh (64 bytes)
		copy(byteData[offset+128:offset+192], block.Qh[:])
		// Copy Scales (16 bytes)
		for j := 0; j < 16; j++ {
			byteData[offset+192+j] = byte(block.Scales[j])
		}
		// Copy D (2 bytes)
		byteData[offset+208] = byte(block.D)
		byteData[offset+209] = byte(block.D >> 8)
	}

	// Create output tensor
	// Compute stride (row-major layout)
	stride := make([]int, len(t.shape))
	if len(t.shape) > 0 {
		stride[len(t.shape)-1] = 1
		for i := len(t.shape) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * t.shape[i+1]
		}
	}

	return &Tensor{
		shape:  t.shape,
		dtype:  Q6_K,
		data:   byteData,
		stride: stride,
		device: CPU,
		offset: 0,
	}, nil
}

// QuantizeQ4_KTensor converts a Float32 tensor to Q4_K format.
// This is a helper function primarily for testing.
func QuantizeQ4_KTensor(t *Tensor) (*Tensor, error) {
	if t.DType() != Float32 {
		return nil, fmt.Errorf("input tensor must be Float32, got %s", t.DType())
	}

	data := t.data.([]float32)

	// Quantize the data
	blocks, err := QuantizeQ4_K(data)
	if err != nil {
		return nil, err
	}

	// Convert blocks to byte array
	blockSize := len(blocks)
	byteData := make([]byte, blockSize*Q4_K_BLOCK_SIZE)

	for i, block := range blocks {
		offset := i * Q4_K_BLOCK_SIZE
		// Copy D (2 bytes)
		byteData[offset] = byte(block.D)
		byteData[offset+1] = byte(block.D >> 8)
		// Copy Dmin (2 bytes)
		byteData[offset+2] = byte(block.Dmin)
		byteData[offset+3] = byte(block.Dmin >> 8)
		// Copy Scales (12 bytes)
		copy(byteData[offset+4:offset+16], block.Scales[:])
		// Copy Qs (128 bytes)
		copy(byteData[offset+16:offset+144], block.Qs[:])
	}

	// Create output tensor
	// Compute stride (row-major layout)
	stride := make([]int, len(t.shape))
	if len(t.shape) > 0 {
		stride[len(t.shape)-1] = 1
		for i := len(t.shape) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * t.shape[i+1]
		}
	}

	return &Tensor{
		shape:  t.shape,
		dtype:  Q4_K,
		data:   byteData,
		stride: stride,
		device: CPU,
		offset: 0,
	}, nil
}
