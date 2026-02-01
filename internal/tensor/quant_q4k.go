package tensor

import (
	"fmt"
	"math"
)

// Q4_K quantization constants
const (
	Q4_K_BLOCK_SIZE = 144 // Total bytes per block: 2 + 2 + 12 + 128
)

// BlockQ4_K represents a Q4_K quantization block.
// Each block contains 256 quantized float32 values compressed to 144 bytes (4.5 bits/weight).
//
// Structure (based on llama.cpp ggml-quants.c):
//   - D: Super-block scale for quantized scales (2 bytes, float16)
//   - Dmin: Super-block scale for quantized mins (2 bytes, float16)
//   - Scales: Scales and mins, quantized with 6 bits (12 bytes)
//   - Qs: 4 bits of quantized values (128 bytes, QK_K/2)
//
// Similar to Q5_K but without the high bit array (Qh), making it more compact.
type BlockQ4_K struct {
	D      uint16     // [2 bytes] Super-block scale (float16)
	Dmin   uint16     // [2 bytes] Super-block min scale (float16)
	Scales [12]uint8  // [12 bytes] Quantized scales and mins (6-bit)
	Qs     [128]uint8 // [128 bytes] 4 bits (2 values per byte)
}

// DequantizeQ4_K converts Q4_K quantized blocks to float32 values.
//
// The Q4_K format uses asymmetric quantization with 4-bit values (0-15):
//   original_value ≈ (scale * quantized_value) - min_offset
//
// Each super-block (256 values) is divided into 8 sub-blocks of 32 values,
// each with its own scale and minimum offset.
func DequantizeQ4_K(blocks []BlockQ4_K, output []float32) error {
	expectedSize := len(blocks) * QK_K
	if len(output) != expectedSize {
		return fmt.Errorf("output size mismatch: got %d, expected %d", len(output), expectedSize)
	}

	for blockIdx, block := range blocks {
		// 1. Convert float16 scale factors to float32
		d := float16ToFloat32(block.D)
		dmin := float16ToFloat32(block.Dmin)

		// 2. Extract scales and mins from the packed Scales array
		sc, m := extractScalesAndMins(block.Scales)

		// 3. Process each of the 8 sub-blocks (32 values each)
		for subBlock := 0; subBlock < 8; subBlock++ {
			// Calculate actual scale and min for this sub-block
			scale := d * float32(sc[subBlock])
			min := dmin * float32(m[subBlock])

			// 4. Dequantize 32 values in this sub-block
			for i := 0; i < 32; i++ {
				valueIdx := subBlock*32 + i

				// 5. Extract 4 bits from Qs (2 values per byte)
				qsIdx := valueIdx / 2
				bits := block.Qs[qsIdx]
				var q uint8
				if valueIdx%2 == 0 {
					q = bits & 0x0F // Lower nibble
				} else {
					q = bits >> 4 // Upper nibble
				}

				// 6. Dequantize: output = scale * q - min
				outputIdx := blockIdx*QK_K + valueIdx
				output[outputIdx] = scale*float32(q) - min
			}
		}
	}

	return nil
}

// DequantizeQ4_KElement dequantizes a single element from a Q4_K tensor.
// This is used for on-demand dequantization in Tensor.At().
//
// Parameters:
//   - data: Raw byte slice containing Q4_K blocks
//   - idx: Global element index (0 to numElements-1)
//
// Returns the dequantized float32 value.
func DequantizeQ4_KElement(data []byte, idx int) float32 {
	// Calculate which block and offset within block
	blockIdx := idx / QK_K
	offsetInBlock := idx % QK_K

	// Extract the block from raw bytes
	blockOffset := blockIdx * Q4_K_BLOCK_SIZE
	if blockOffset+Q4_K_BLOCK_SIZE > len(data) {
		// Out of bounds - return 0
		return 0
	}

	// Parse block structure from bytes
	block := parseQ4_KBlock(data[blockOffset : blockOffset+Q4_K_BLOCK_SIZE])

	// Convert float16 scale factors
	d := float16ToFloat32(block.D)
	dmin := float16ToFloat32(block.Dmin)

	// Extract scales and mins
	sc, m := extractScalesAndMins(block.Scales)

	// Determine which sub-block (0-7) this element is in
	subBlock := offsetInBlock / 32
	valueIdx := offsetInBlock % 32

	// Calculate scale and min for this sub-block
	scale := d * float32(sc[subBlock])
	min := dmin * float32(m[subBlock])

	// Get absolute value index within the block
	absIdx := subBlock*32 + valueIdx

	// Extract 4 bits
	qsIdx := absIdx / 2
	bits := block.Qs[qsIdx]
	var q uint8
	if absIdx%2 == 0 {
		q = bits & 0x0F
	} else {
		q = bits >> 4
	}

	// Dequantize
	return scale*float32(q) - min
}

// parseQ4_KBlock parses a BlockQ4_K from raw bytes.
func parseQ4_KBlock(data []byte) BlockQ4_K {
	if len(data) < Q4_K_BLOCK_SIZE {
		return BlockQ4_K{}
	}

	block := BlockQ4_K{}

	// Parse D (2 bytes, little-endian uint16 representing float16)
	block.D = uint16(data[0]) | uint16(data[1])<<8

	// Parse Dmin (2 bytes)
	block.Dmin = uint16(data[2]) | uint16(data[3])<<8

	// Parse Scales (12 bytes)
	copy(block.Scales[:], data[4:16])

	// Parse Qs (128 bytes)
	copy(block.Qs[:], data[16:144])

	return block
}

// DequantizeQ4_KTensor dequantizes an entire Q4_K tensor to Float32.
// This is useful for eager dequantization of embeddings or frequently accessed tensors.
func DequantizeQ4_KTensor(t *Tensor) (*Tensor, error) {
	if t.dtype != Q4_K {
		return nil, fmt.Errorf("tensor is not Q4_K, got %s", t.dtype)
	}

	// Calculate number of blocks based on actual data size, not shape
	// The shape might be incorrect for Q4_K tensors loaded from GGUF
	rawData := t.data.([]byte)
	numBlocks := len(rawData) / Q4_K_BLOCK_SIZE
	numElements := numBlocks * QK_K

	// Parse blocks from raw bytes
	blocks := make([]BlockQ4_K, numBlocks)
	for i := 0; i < numBlocks; i++ {
		offset := i * Q4_K_BLOCK_SIZE
		if offset+Q4_K_BLOCK_SIZE <= len(rawData) {
			blocks[i] = parseQ4_KBlock(rawData[offset : offset+Q4_K_BLOCK_SIZE])
		}
	}

	// Allocate output
	output := make([]float32, numElements)

	// Dequantize
	if err := DequantizeQ4_K(blocks, output); err != nil {
		return nil, fmt.Errorf("dequantization failed: %w", err)
	}

	// Create new Float32 tensor with same shape
	// IMPORTANT: Must compute stride for the new tensor!
	result := &Tensor{
		shape:  t.shape,
		stride: computeStrides(t.shape),
		dtype:  Float32,
		data:   output,
		device: CPU,
		offset: 0,
	}

	// Verify the result
	resultSize := 1
	for _, dim := range result.shape {
		resultSize *= dim
	}
	if resultSize != len(output) {
		fmt.Printf("WARNING: Shape mismatch! shape=%v implies %d elements, but data has %d elements\n",
			result.shape, resultSize, len(output))
	}

	return result, nil
}

// QuantizeQ4_K quantizes float32 values to Q4_K format.
// This is primarily used for testing and validation.
func QuantizeQ4_K(input []float32) ([]BlockQ4_K, error) {
	numElements := len(input)
	numBlocks := (numElements + QK_K - 1) / QK_K
	blocks := make([]BlockQ4_K, numBlocks)

	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		block := &blocks[blockIdx]
		offset := blockIdx * QK_K

		// Find min and max for this block to determine scale
		min := float32(math.MaxFloat32)
		max := float32(-math.MaxFloat32)

		endIdx := offset + QK_K
		if endIdx > numElements {
			endIdx = numElements
		}

		for i := offset; i < endIdx; i++ {
			val := input[i]
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}

		// Calculate scale (range / 15, since we have 4 bits = 16 levels)
		scale := (max - min) / 15.0
		if scale < 1e-10 {
			scale = 1e-10 // Avoid division by zero
		}

		// Simplified quantization (not optimized, just for testing)
		// In practice, Q4_K uses more sophisticated quantization with sub-blocks
		block.D = float32ToFloat16(scale)
		block.Dmin = float32ToFloat16(min)

		// For testing purposes, use simplified sub-block scales
		// All sub-blocks get same scale/min
		for i := 0; i < 8; i++ {
			block.Scales[0] = 32 // Middle value for 6-bit (0-63)
			block.Scales[6] = 32
		}

		// Quantize values
		for i := offset; i < endIdx; i++ {
			idx := i - offset

			// Quantize to 4 bits (0-15)
			q := uint8((input[i] - min) / scale)
			if q > 15 {
				q = 15
			}

			// Pack 4 bits into Qs (2 values per byte)
			qsIdx := idx / 2
			if idx%2 == 0 {
				block.Qs[qsIdx] = q & 0x0F
			} else {
				block.Qs[qsIdx] |= (q & 0x0F) << 4
			}
		}
	}

	return blocks, nil
}
