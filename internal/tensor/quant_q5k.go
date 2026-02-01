package tensor

import (
	"fmt"
	"math"
)

// Q5_K quantization constants
const (
	QK_K           = 256 // Elements per super-block
	K_SCALE_SIZE   = 12  // Number of scale bytes
	Q5_K_BLOCK_SIZE = 176 // Total bytes per block: 2 + 2 + 12 + 32 + 128
)

// BlockQ5_K represents a Q5_K quantization block.
// Each block contains 256 quantized float32 values compressed to 176 bytes (5.5 bits/weight).
//
// Structure:
//   - D: Super-block scale for quantized scales (2 bytes, float16)
//   - Dmin: Super-block scale for quantized mins (2 bytes, float16)
//   - Scales: Scales and mins, quantized with 6 bits (12 bytes)
//   - Qh: High bits of quantized values (32 bytes, QK_K/8)
//   - Qs: Low 4 bits of quantized values (128 bytes, QK_K/2)
type BlockQ5_K struct {
	D      uint16    // [2 bytes] Super-block scale (float16)
	Dmin   uint16    // [2 bytes] Super-block min scale (float16)
	Scales [12]uint8 // [12 bytes] Quantized scales and mins (6-bit)
	Qh     [32]uint8 // [32 bytes] High bits (5th bit of each value)
	Qs     [128]uint8 // [128 bytes] Low 4 bits (2 values per byte)
}

// DequantizeQ5_K converts Q5_K quantized blocks to float32 values.
//
// The Q5_K format uses asymmetric quantization with 5-bit values (0-31):
//   original_value ≈ (scale * quantized_value) - min_offset
//
// Each super-block (256 values) is divided into 8 sub-blocks of 32 values,
// each with its own scale and minimum offset.
func DequantizeQ5_K(blocks []BlockQ5_K, output []float32) error {
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

				// 5. Extract low 4 bits from Qs (2 values per byte)
				qsIdx := valueIdx / 2
				lowBits := block.Qs[qsIdx]
				if valueIdx%2 == 0 {
					lowBits &= 0x0F // Lower nibble
				} else {
					lowBits >>= 4 // Upper nibble
				}

				// 6. Extract high bit from Qh (8 values per byte)
				qhIdx := valueIdx / 8
				bitPos := valueIdx % 8
				highBit := (block.Qh[qhIdx] >> bitPos) & 0x01

				// 7. Combine to form 5-bit quantized value (0-31)
				q := uint8(lowBits) | (highBit << 4)

				// 8. Dequantize: output = scale * q - min
				outputIdx := blockIdx*QK_K + valueIdx
				output[outputIdx] = scale*float32(q) - min
			}
		}
	}

	return nil
}

// extractScalesAndMins unpacks 6-bit quantized scales and mins from the 12-byte Scales array.
//
// The 12 bytes encode 8 pairs of 6-bit values:
//   - Bytes 0-5: 8 scales (6 bits each)
//   - Bytes 6-11: 8 mins (6 bits each)
//
// Each 6-bit value is packed tightly, requiring bit shifting and masking.
func extractScalesAndMins(scales [12]uint8) (sc [8]uint8, m [8]uint8) {
	// Extract 8 scales from first 6 bytes (4 scales per 3 bytes)
	// Pattern: 6 bits from byte N, possibly spanning into byte N+1
	sc[0] = scales[0] & 0x3F
	sc[1] = (scales[0]>>6 | scales[1]<<2) & 0x3F
	sc[2] = (scales[1]>>4 | scales[2]<<4) & 0x3F
	sc[3] = (scales[2] >> 2) & 0x3F
	sc[4] = scales[3] & 0x3F
	sc[5] = (scales[3]>>6 | scales[4]<<2) & 0x3F
	sc[6] = (scales[4]>>4 | scales[5]<<4) & 0x3F
	sc[7] = (scales[5] >> 2) & 0x3F

	// Extract 8 mins from next 6 bytes (same packing scheme)
	m[0] = scales[6] & 0x3F
	m[1] = (scales[6]>>6 | scales[7]<<2) & 0x3F
	m[2] = (scales[7]>>4 | scales[8]<<4) & 0x3F
	m[3] = (scales[8] >> 2) & 0x3F
	m[4] = scales[9] & 0x3F
	m[5] = (scales[9]>>6 | scales[10]<<2) & 0x3F
	m[6] = (scales[10]>>4 | scales[11]<<4) & 0x3F
	m[7] = (scales[11] >> 2) & 0x3F

	return sc, m
}

// DequantizeQ5_KElement dequantizes a single element from a Q5_K tensor.
// This is used for on-demand dequantization in Tensor.At().
//
// Parameters:
//   - data: Raw byte slice containing Q5_K blocks
//   - idx: Global element index (0 to numElements-1)
//
// Returns the dequantized float32 value.
func DequantizeQ5_KElement(data []byte, idx int) float32 {
	// Calculate which block and offset within block
	blockIdx := idx / QK_K
	offsetInBlock := idx % QK_K

	// Extract the block from raw bytes
	blockOffset := blockIdx * Q5_K_BLOCK_SIZE
	if blockOffset+Q5_K_BLOCK_SIZE > len(data) {
		// Out of bounds - return 0
		return 0
	}

	// Parse block structure from bytes
	block := parseQ5_KBlock(data[blockOffset : blockOffset+Q5_K_BLOCK_SIZE])

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

	// Extract low 4 bits
	qsIdx := absIdx / 2
	lowBits := block.Qs[qsIdx]
	if absIdx%2 == 0 {
		lowBits &= 0x0F
	} else {
		lowBits >>= 4
	}

	// Extract high bit
	qhIdx := absIdx / 8
	bitPos := absIdx % 8
	highBit := (block.Qh[qhIdx] >> bitPos) & 0x01

	// Combine to 5-bit value
	q := uint8(lowBits) | (highBit << 4)

	// Dequantize
	return scale*float32(q) - min
}

// parseQ5_KBlock parses a BlockQ5_K from raw bytes.
func parseQ5_KBlock(data []byte) BlockQ5_K {
	if len(data) < Q5_K_BLOCK_SIZE {
		return BlockQ5_K{}
	}

	block := BlockQ5_K{}

	// Parse D (2 bytes, little-endian uint16 representing float16)
	block.D = uint16(data[0]) | uint16(data[1])<<8

	// Parse Dmin (2 bytes)
	block.Dmin = uint16(data[2]) | uint16(data[3])<<8

	// Parse Scales (12 bytes)
	copy(block.Scales[:], data[4:16])

	// Parse Qh (32 bytes)
	copy(block.Qh[:], data[16:48])

	// Parse Qs (128 bytes)
	copy(block.Qs[:], data[48:176])

	return block
}

// DequantizeQ5_KTensor dequantizes an entire Q5_K tensor to Float32.
// This is useful for eager dequantization of embeddings or frequently accessed tensors.
func DequantizeQ5_KTensor(t *Tensor) (*Tensor, error) {
	if t.dtype != Q5_K {
		return nil, fmt.Errorf("tensor is not Q5_K, got %s", t.dtype)
	}

	// Calculate number of blocks based on actual data size, not shape
	// The shape might be incorrect for Q5_K tensors loaded from GGUF
	rawData := t.data.([]byte)
	numBlocks := len(rawData) / Q5_K_BLOCK_SIZE
	numElements := numBlocks * QK_K

	// Parse blocks from raw bytes
	blocks := make([]BlockQ5_K, numBlocks)
	for i := 0; i < numBlocks; i++ {
		offset := i * Q5_K_BLOCK_SIZE
		if offset+Q5_K_BLOCK_SIZE <= len(rawData) {
			blocks[i] = parseQ5_KBlock(rawData[offset : offset+Q5_K_BLOCK_SIZE])
		}
	}

	// Allocate output
	output := make([]float32, numElements)

	// Dequantize
	if err := DequantizeQ5_K(blocks, output); err != nil {
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

// QuantizeQ5_K quantizes float32 values to Q5_K format.
// This is primarily used for testing and validation.
func QuantizeQ5_K(input []float32) ([]BlockQ5_K, error) {
	numElements := len(input)
	numBlocks := (numElements + QK_K - 1) / QK_K
	blocks := make([]BlockQ5_K, numBlocks)

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

		// Calculate scale (range / 31, since we have 5 bits = 32 levels)
		scale := (max - min) / 31.0
		if scale < 1e-10 {
			scale = 1e-10 // Avoid division by zero
		}

		// Simplified quantization (not optimized, just for testing)
		// In practice, Q5_K uses more sophisticated quantization with sub-blocks
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

			// Quantize to 5 bits (0-31)
			q := uint8((input[i] - min) / scale)
			if q > 31 {
				q = 31
			}

			// Pack low 4 bits into Qs
			qsIdx := idx / 2
			if idx%2 == 0 {
				block.Qs[qsIdx] = q & 0x0F
			} else {
				block.Qs[qsIdx] |= (q & 0x0F) << 4
			}

			// Pack high bit into Qh
			qhIdx := idx / 8
			bitPos := idx % 8
			if (q & 0x10) != 0 {
				block.Qh[qhIdx] |= 1 << bitPos
			}
		}
	}

	return blocks, nil
}

