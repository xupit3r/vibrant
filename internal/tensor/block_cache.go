package tensor

import (
	"sync"
)

// Q5_K block cache for efficient batch dequantization
// Instead of dequantizing element-by-element, we dequantize entire blocks (256 elements)
// and cache them for reuse across multiple matrix elements.

const (
	Q5K_BLOCK_ELEMENTS = 256 // QK_K from quant_q5k.go
)

// BlockCache provides efficient caching of dequantized Q5_K blocks
type BlockCache struct {
	blocks map[int][]float32 // blockIdx -> dequantized 256 elements
	mu     sync.RWMutex
}

// NewBlockCache creates a new block cache
func NewBlockCache() *BlockCache {
	return &BlockCache{
		blocks: make(map[int][]float32),
	}
}

// GetOrDequantizeQ5K retrieves a dequantized block or dequantizes it on-demand
func (bc *BlockCache) GetOrDequantizeQ5K(bData []byte, blockIdx int) []float32 {
	// Fast path: Check if already cached (read lock)
	bc.mu.RLock()
	if block, exists := bc.blocks[blockIdx]; exists {
		bc.mu.RUnlock()
		return block
	}
	bc.mu.RUnlock()

	// Slow path: Dequantize block (write lock)
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Double-check after acquiring write lock (another goroutine might have added it)
	if block, exists := bc.blocks[blockIdx]; exists {
		return block
	}

	// Dequantize the entire block
	block := make([]float32, Q5K_BLOCK_ELEMENTS)
	blockOffset := blockIdx * Q5_K_BLOCK_SIZE

	if blockOffset+Q5_K_BLOCK_SIZE > len(bData) {
		// Out of bounds - return zeros
		bc.blocks[blockIdx] = block
		return block
	}

	// Parse Q5_K block
	q5kBlock := parseQ5_KBlock(bData[blockOffset : blockOffset+Q5_K_BLOCK_SIZE])

	// Dequantize all 256 elements at once
	dequantizeQ5KBlockBatch(q5kBlock, block)

	// Cache for future use
	bc.blocks[blockIdx] = block

	return block
}

// Clear removes all cached blocks
func (bc *BlockCache) Clear() {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	bc.blocks = make(map[int][]float32)
}

// dequantizeQ5KBlockBatch dequantizes an entire Q5_K block into a float32 array
// This is much faster than calling DequantizeQ5_KElement 256 times because:
// 1. Parse block structure once (not 256 times)
// 2. Extract scales/mins once (not 256 times)
// 3. Better cache locality (sequential access)
// 4. Can be SIMD-optimized later
func dequantizeQ5KBlockBatch(block BlockQ5_K, output []float32) {
	// 1. Convert float16 scale factors to float32
	d := float16ToFloat32(block.D)
	dmin := float16ToFloat32(block.Dmin)

	// 2. Extract scales and mins from the packed Scales array (once for all 256 elements)
	sc, m := extractScalesAndMins(block.Scales)

	// 3. Process each of the 8 sub-blocks (32 values each)
	for subBlock := 0; subBlock < 8; subBlock++ {
		// Calculate actual scale and min for this sub-block
		scale := d * float32(sc[subBlock])
		min := dmin * float32(m[subBlock])

		// 4. Dequantize 32 values in this sub-block
		baseIdx := subBlock * 32
		for i := 0; i < 32; i++ {
			valueIdx := baseIdx + i

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
			output[valueIdx] = scale*float32(q) - min
		}
	}
}

// Q6_K block cache

// GetOrDequantizeQ6K retrieves a dequantized Q6_K block or dequantizes it on-demand
func (bc *BlockCache) GetOrDequantizeQ6K(bData []byte, blockIdx int) []float32 {
	bc.mu.RLock()
	if block, exists := bc.blocks[blockIdx]; exists {
		bc.mu.RUnlock()
		return block
	}
	bc.mu.RUnlock()

	bc.mu.Lock()
	defer bc.mu.Unlock()

	if block, exists := bc.blocks[blockIdx]; exists {
		return block
	}

	block := make([]float32, Q5K_BLOCK_ELEMENTS) // Q6_K also uses 256 elements
	blockOffset := blockIdx * Q6_K_BLOCK_SIZE

	if blockOffset+Q6_K_BLOCK_SIZE > len(bData) {
		bc.blocks[blockIdx] = block
		return block
	}

	q6kBlock := parseQ6_KBlock(bData[blockOffset : blockOffset+Q6_K_BLOCK_SIZE])
	dequantizeQ6KBlockBatch(q6kBlock, block)
	bc.blocks[blockIdx] = block

	return block
}

// dequantizeQ6KBlockBatch dequantizes an entire Q6_K block
func dequantizeQ6KBlockBatch(block BlockQ6_K, output []float32) {
	d := float16ToFloat32(block.D)

	for i := 0; i < Q5K_BLOCK_ELEMENTS; i++ {
		// Extract low 4 bits from Ql
		qlIdx := i / 2
		lowBits := block.Ql[qlIdx]
		if i%2 == 0 {
			lowBits &= 0x0F
		} else {
			lowBits >>= 4
		}

		// Extract high 2 bits from Qh
		qhIdx := i / 4
		bitPos := (i % 4) * 2
		highBits := (block.Qh[qhIdx] >> bitPos) & 0x03

		// Combine to 6-bit value
		q := uint8(lowBits) | (highBits << 4)

		// Get scale for this group
		scaleIdx := i / 16
		scale := float32(block.Scales[scaleIdx])

		// Dequantize
		output[i] = d * scale * float32(int(q)-32)
	}
}

// Q4_K block cache

// GetOrDequantizeQ4K retrieves a dequantized Q4_K block or dequantizes it on-demand
func (bc *BlockCache) GetOrDequantizeQ4K(bData []byte, blockIdx int) []float32 {
	bc.mu.RLock()
	if block, exists := bc.blocks[blockIdx]; exists {
		bc.mu.RUnlock()
		return block
	}
	bc.mu.RUnlock()

	bc.mu.Lock()
	defer bc.mu.Unlock()

	if block, exists := bc.blocks[blockIdx]; exists {
		return block
	}

	block := make([]float32, Q5K_BLOCK_ELEMENTS) // Q4_K also uses 256 elements
	blockOffset := blockIdx * Q4_K_BLOCK_SIZE

	if blockOffset+Q4_K_BLOCK_SIZE > len(bData) {
		bc.blocks[blockIdx] = block
		return block
	}

	q4kBlock := parseQ4_KBlock(bData[blockOffset : blockOffset+Q4_K_BLOCK_SIZE])
	dequantizeQ4KBlockBatch(q4kBlock, block)
	bc.blocks[blockIdx] = block

	return block
}

// dequantizeQ4KBlockBatch dequantizes an entire Q4_K block
func dequantizeQ4KBlockBatch(block BlockQ4_K, output []float32) {
	d := float16ToFloat32(block.D)
	dmin := float16ToFloat32(block.Dmin)

	sc, m := extractScalesAndMins(block.Scales)

	for subBlock := 0; subBlock < 8; subBlock++ {
		scale := d * float32(sc[subBlock])
		min := dmin * float32(m[subBlock])

		baseIdx := subBlock * 32
		for i := 0; i < 32; i++ {
			valueIdx := baseIdx + i

			qsIdx := valueIdx / 2
			bits := block.Qs[qsIdx]
			var q uint8
			if valueIdx%2 == 0 {
				q = bits & 0x0F
			} else {
				q = bits >> 4
			}

			output[valueIdx] = scale*float32(q) - min
		}
	}
}
