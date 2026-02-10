package tensor

import "fmt"

// Fused dequantize-transpose functions.
//
// Standard path: dequantize → alloc M*N → transpose → alloc N*M → discard first.
// Fused path:    alloc N*M → dequantize each block → scatter-write to transposed positions.
//
// This halves peak memory per weight (one allocation instead of two transient ones)
// while producing the same result.

// DequantTransposeQ4K dequantizes a 2D Q4_K tensor and writes directly
// into transposed layout. Returns a [N, M] Float32 tensor marked as transposed.
func DequantTransposeQ4K(t *Tensor) (*Tensor, error) {
	if t.dtype != Q4_K {
		return nil, fmt.Errorf("expected Q4_K tensor, got %s", t.dtype)
	}
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor, got shape %v", t.shape)
	}

	M, N := t.shape[0], t.shape[1]
	rawData := t.data.([]byte)
	totalElements := M * N
	numBlocks := totalElements / QK_K

	// Single allocation: transposed output [N, M]
	out := make([]float32, totalElements)

	// Small buffer for one block (256 floats = 1KB on stack)
	var buf [QK_K]float32

	for bi := 0; bi < numBlocks; bi++ {
		blockOffset := bi * Q4_K_BLOCK_SIZE
		if blockOffset+Q4_K_BLOCK_SIZE > len(rawData) {
			break
		}

		block := parseQ4_KBlock(rawData[blockOffset : blockOffset+Q4_K_BLOCK_SIZE])
		dequantizeQ4KBlockBatch(block, buf[:])

		// Scatter-write to transposed positions
		globalStart := bi * QK_K
		for k := 0; k < QK_K; k++ {
			srcIdx := globalStart + k
			srcRow := srcIdx / N
			srcCol := srcIdx % N
			// Transposed position: [col, row] in [N, M] layout
			out[srcCol*M+srcRow] = buf[k]
		}
	}

	result := &Tensor{
		data:       out,
		shape:      []int{N, M},
		stride:     computeStrides([]int{N, M}),
		dtype:      Float32,
		device:     CPU,
		transposed: true,
	}
	return result, nil
}

// DequantTransposeQ5K dequantizes a 2D Q5_K tensor and writes directly
// into transposed layout. Returns a [N, M] Float32 tensor marked as transposed.
func DequantTransposeQ5K(t *Tensor) (*Tensor, error) {
	if t.dtype != Q5_K {
		return nil, fmt.Errorf("expected Q5_K tensor, got %s", t.dtype)
	}
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor, got shape %v", t.shape)
	}

	M, N := t.shape[0], t.shape[1]
	rawData := t.data.([]byte)
	totalElements := M * N
	numBlocks := totalElements / QK_K

	out := make([]float32, totalElements)
	var buf [QK_K]float32

	for bi := 0; bi < numBlocks; bi++ {
		blockOffset := bi * Q5_K_BLOCK_SIZE
		if blockOffset+Q5_K_BLOCK_SIZE > len(rawData) {
			break
		}

		block := parseQ5_KBlock(rawData[blockOffset : blockOffset+Q5_K_BLOCK_SIZE])
		dequantizeQ5KBlockBatch(block, buf[:])

		globalStart := bi * QK_K
		for k := 0; k < QK_K; k++ {
			srcIdx := globalStart + k
			srcRow := srcIdx / N
			srcCol := srcIdx % N
			out[srcCol*M+srcRow] = buf[k]
		}
	}

	result := &Tensor{
		data:       out,
		shape:      []int{N, M},
		stride:     computeStrides([]int{N, M}),
		dtype:      Float32,
		device:     CPU,
		transposed: true,
	}
	return result, nil
}

// DequantTransposeQ6K dequantizes a 2D Q6_K tensor and writes directly
// into transposed layout. Returns a [N, M] Float32 tensor marked as transposed.
func DequantTransposeQ6K(t *Tensor) (*Tensor, error) {
	if t.dtype != Q6_K {
		return nil, fmt.Errorf("expected Q6_K tensor, got %s", t.dtype)
	}
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor, got shape %v", t.shape)
	}

	M, N := t.shape[0], t.shape[1]
	rawData := t.data.([]byte)
	totalElements := M * N
	numBlocks := totalElements / QK_K

	out := make([]float32, totalElements)
	var buf [QK_K]float32

	for bi := 0; bi < numBlocks; bi++ {
		blockOffset := bi * Q6_K_BLOCK_SIZE
		if blockOffset+Q6_K_BLOCK_SIZE > len(rawData) {
			break
		}

		block := parseQ6_KBlock(rawData[blockOffset : blockOffset+Q6_K_BLOCK_SIZE])
		dequantizeQ6KBlockBatch(block, buf[:])

		globalStart := bi * QK_K
		for k := 0; k < QK_K; k++ {
			srcIdx := globalStart + k
			srcRow := srcIdx / N
			srcCol := srcIdx % N
			out[srcCol*M+srcRow] = buf[k]
		}
	}

	result := &Tensor{
		data:       out,
		shape:      []int{N, M},
		stride:     computeStrides([]int{N, M}),
		dtype:      Float32,
		device:     CPU,
		transposed: true,
	}
	return result, nil
}
