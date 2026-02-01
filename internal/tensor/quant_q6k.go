package tensor

import (
	"fmt"
	"math"
)

// Q6_K quantization constants
const (
	Q6_K_BLOCK_SIZE = 210 // Total bytes per block: 128 + 64 + 16 + 2
)

// BlockQ6_K represents a Q6_K quantization block.
// Each block contains 256 quantized float32 values compressed to 210 bytes (6.5625 bits/weight).
//
// Structure (based on llama.cpp ggml-quants.c):
//   - Ql: Low 4 bits of quantized values (128 bytes, QK_K/2)
//   - Qh: High 2 bits of quantized values (64 bytes, QK_K/4)
//   - Scales: Scales for each group of 16 values (16 bytes, int8)
//   - D: Super-block scale (2 bytes, float16)
//
// Each value is quantized to 6 bits (0-63), then scaled:
//   original_value ≈ d * scale[i/16] * (q - 32)
// where q is the 6-bit quantized value formed by combining low and high bits.
type BlockQ6_K struct {
	Ql     [128]uint8 // [128 bytes] Low 4 bits (2 values per byte)
	Qh     [64]uint8  // [64 bytes] High 2 bits (4 values per byte)
	Scales [16]int8   // [16 bytes] Scales (one per 16 values)
	D      uint16     // [2 bytes] Super-block scale (float16)
}

// DequantizeQ6_K converts Q6_K quantized blocks to float32 values.
//
// The Q6_K format uses symmetric quantization with 6-bit values (0-63):
//   original_value ≈ d * scale * (quantized_value - 32)
//
// Each super-block (256 values) is divided into 16 groups of 16 values,
// each with its own int8 scale factor.
func DequantizeQ6_K(blocks []BlockQ6_K, output []float32) error {
	expectedSize := len(blocks) * QK_K
	if len(output) != expectedSize {
		return fmt.Errorf("output size mismatch: got %d, expected %d", len(output), expectedSize)
	}

	for blockIdx, block := range blocks {
		// 1. Convert float16 super-block scale to float32
		d := float16ToFloat32(block.D)

		// 2. Dequantize all 256 values in this block
		for i := 0; i < QK_K; i++ {
			// 3. Extract low 4 bits from Ql (2 values per byte)
			qlIdx := i / 2
			lowBits := block.Ql[qlIdx]
			if i%2 == 0 {
				lowBits &= 0x0F // Lower nibble
			} else {
				lowBits >>= 4 // Upper nibble
			}

			// 4. Extract high 2 bits from Qh (4 values per byte)
			qhIdx := i / 4
			bitPos := (i % 4) * 2
			highBits := (block.Qh[qhIdx] >> bitPos) & 0x03

			// 5. Combine to form 6-bit quantized value (0-63)
			q := uint8(lowBits) | (highBits << 4)

			// 6. Get scale for this group (one scale per 16 values)
			scaleIdx := i / 16
			scale := float32(block.Scales[scaleIdx])

			// 7. Dequantize: output = d * scale * (q - 32)
			// Subtract 32 to center the range around 0
			outputIdx := blockIdx*QK_K + i
			output[outputIdx] = d * scale * float32(int(q)-32)
		}
	}

	return nil
}

// DequantizeQ6_KElement dequantizes a single element from a Q6_K tensor.
// This is used for on-demand dequantization in Tensor.At().
//
// Parameters:
//   - data: Raw byte slice containing Q6_K blocks
//   - idx: Global element index (0 to numElements-1)
//
// Returns the dequantized float32 value.
func DequantizeQ6_KElement(data []byte, idx int) float32 {
	// Calculate which block and offset within block
	blockIdx := idx / QK_K
	offsetInBlock := idx % QK_K

	// Extract the block from raw bytes
	blockOffset := blockIdx * Q6_K_BLOCK_SIZE
	if blockOffset+Q6_K_BLOCK_SIZE > len(data) {
		// Out of bounds - return 0
		return 0
	}

	// Parse block structure from bytes
	block := parseQ6_KBlock(data[blockOffset : blockOffset+Q6_K_BLOCK_SIZE])

	// Convert float16 super-block scale
	d := float16ToFloat32(block.D)

	// Extract low 4 bits
	qlIdx := offsetInBlock / 2
	lowBits := block.Ql[qlIdx]
	if offsetInBlock%2 == 0 {
		lowBits &= 0x0F
	} else {
		lowBits >>= 4
	}

	// Extract high 2 bits
	qhIdx := offsetInBlock / 4
	bitPos := (offsetInBlock % 4) * 2
	highBits := (block.Qh[qhIdx] >> bitPos) & 0x03

	// Combine to 6-bit value
	q := uint8(lowBits) | (highBits << 4)

	// Get scale for this group
	scaleIdx := offsetInBlock / 16
	scale := float32(block.Scales[scaleIdx])

	// Dequantize
	return d * scale * float32(int(q)-32)
}

// parseQ6_KBlock parses a BlockQ6_K from raw bytes.
func parseQ6_KBlock(data []byte) BlockQ6_K {
	if len(data) < Q6_K_BLOCK_SIZE {
		return BlockQ6_K{}
	}

	block := BlockQ6_K{}

	// Parse Ql (128 bytes, starting at offset 0)
	copy(block.Ql[:], data[0:128])

	// Parse Qh (64 bytes, starting at offset 128)
	copy(block.Qh[:], data[128:192])

	// Parse Scales (16 bytes, starting at offset 192)
	for i := 0; i < 16; i++ {
		block.Scales[i] = int8(data[192+i])
	}

	// Parse D (2 bytes, starting at offset 208, little-endian)
	block.D = uint16(data[208]) | uint16(data[209])<<8

	return block
}

// DequantizeQ6_KTensor dequantizes an entire Q6_K tensor to Float32.
// This is useful for eager dequantization of embeddings or frequently accessed tensors.
func DequantizeQ6_KTensor(t *Tensor) (*Tensor, error) {
	if t.dtype != Q6_K {
		return nil, fmt.Errorf("tensor is not Q6_K, got %s", t.dtype)
	}

	// Calculate number of blocks based on actual data size, not shape
	// The shape might be incorrect for Q6_K tensors loaded from GGUF
	rawData := t.data.([]byte)
	numBlocks := len(rawData) / Q6_K_BLOCK_SIZE
	numElements := numBlocks * QK_K

	// Parse blocks from raw bytes
	blocks := make([]BlockQ6_K, numBlocks)
	for i := 0; i < numBlocks; i++ {
		offset := i * Q6_K_BLOCK_SIZE
		if offset+Q6_K_BLOCK_SIZE <= len(rawData) {
			blocks[i] = parseQ6_KBlock(rawData[offset : offset+Q6_K_BLOCK_SIZE])
		}
	}

	// Allocate output
	output := make([]float32, numElements)

	// Dequantize
	if err := DequantizeQ6_K(blocks, output); err != nil {
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

// QuantizeQ6_K quantizes float32 values to Q6_K format.
// This is primarily used for testing and validation.
func QuantizeQ6_K(input []float32) ([]BlockQ6_K, error) {
	numElements := len(input)
	numBlocks := (numElements + QK_K - 1) / QK_K
	blocks := make([]BlockQ6_K, numBlocks)

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

		// Calculate super-block scale
		// 6 bits gives us range -32 to +31 (after subtracting 32)
		absMax := float32(math.Max(math.Abs(float64(max)), math.Abs(float64(min))))
		d := absMax / 32.0 // Scale to fit in ±32 range
		if d < 1e-10 {
			d = 1e-10 // Avoid division by zero
		}

		block.D = float32ToFloat16(d)

		// For testing purposes, use simplified group scales
		// In practice, Q6_K uses more sophisticated per-group quantization
		for i := 0; i < 16; i++ {
			block.Scales[i] = 1 // Unity scale for simplicity
		}

		// Quantize values
		for i := offset; i < endIdx; i++ {
			idx := i - offset

			// Determine group and get scale
			groupIdx := idx / 16
			scale := float32(block.Scales[groupIdx])

			// Quantize to 6 bits (0-63)
			// First scale by d and group scale, then add 32 offset
			scaled := input[i] / (d * scale)
			q := int(scaled + 32.5) // Round to nearest
			if q < 0 {
				q = 0
			}
			if q > 63 {
				q = 63
			}

			// Pack low 4 bits into Ql
			qlIdx := idx / 2
			if idx%2 == 0 {
				block.Ql[qlIdx] = uint8(q & 0x0F)
			} else {
				block.Ql[qlIdx] |= uint8((q & 0x0F) << 4)
			}

			// Pack high 2 bits into Qh
			qhIdx := idx / 4
			bitPos := (idx % 4) * 2
			highBits := uint8((q >> 4) & 0x03)
			block.Qh[qhIdx] |= highBits << bitPos
		}
	}

	return blocks, nil
}
