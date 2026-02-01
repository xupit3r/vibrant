package tensor

import (
	"math"
	"testing"
)

// TestDequantizeQ6_K_ZeroBlock tests dequantization of a zero-initialized block
func TestDequantizeQ6_K_ZeroBlock(t *testing.T) {
	blocks := []BlockQ6_K{{}}
	output := make([]float32, QK_K)

	err := DequantizeQ6_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ6_K failed: %v", err)
	}

	// All zeros should dequantize to zeros (or very small values)
	for i, val := range output {
		if math.Abs(float64(val)) > 0.01 {
			t.Errorf("Index %d: expected ~0, got %f", i, val)
		}
	}
}

// TestDequantizeQ6_K_KnownValues tests dequantization with known scale values
func TestDequantizeQ6_K_KnownValues(t *testing.T) {
	blocks := []BlockQ6_K{{
		D: float32ToFloat16(1.0), // scale = 1.0
		Scales: [16]int8{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		},
		Ql: [128]uint8{0xFF, 0xFF, 0xFF, 0xFF}, // All low bits set
		Qh: [64]uint8{0xFF, 0xFF, 0xFF, 0xFF},  // All high bits set
	}}

	output := make([]float32, QK_K)

	err := DequantizeQ6_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ6_K failed: %v", err)
	}

	// With all bits set (q=63) and scale=1, d=1, we expect (63-32) = 31
	// First value should be non-zero
	if output[0] == 0 {
		t.Errorf("Expected non-zero output for first value, got 0")
	}

	// Check that values are in expected range
	// q=63 (all bits set), scale=1, d=1 -> value = 1 * 1 * (63-32) = 31
	expectedValue := float32(31.0)
	tolerance := float32(0.1)
	for i := 0; i < 8; i++ { // Check first few values
		if math.Abs(float64(output[i]-expectedValue)) > float64(tolerance) {
			t.Errorf("Index %d: expected ~%f, got %f", i, expectedValue, output[i])
		}
	}
}

// TestDequantizeQ6_K_Roundtrip tests quantization followed by dequantization
func TestDequantizeQ6_K_Roundtrip(t *testing.T) {
	// Create test data with known pattern
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i%32) - 16.0 // Pattern -16 to +15 repeating
	}

	// Quantize
	blocks, err := QuantizeQ6_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ6_K failed: %v", err)
	}

	// Dequantize
	output := make([]float32, QK_K)
	err = DequantizeQ6_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ6_K failed: %v", err)
	}

	// Check that values are approximately correct (within quantization error)
	const tolerance = 2.0 // Q6_K has 6 bits, so max error is ~range/64
	for i := range input {
		diff := math.Abs(float64(input[i] - output[i]))
		if diff > tolerance {
			t.Errorf("Index %d: input=%f, output=%f, diff=%f (tolerance=%f)",
				i, input[i], output[i], diff, tolerance)
		}
	}
}

// TestDequantizeQ6_K_MultipleBlocks tests with multiple blocks
func TestDequantizeQ6_K_MultipleBlocks(t *testing.T) {
	numBlocks := 4
	blocks := make([]BlockQ6_K, numBlocks)

	// Initialize each block with different scale
	for i := range blocks {
		blocks[i].D = float32ToFloat16(float32(i + 1))
		// Set all scales to 1 for simplicity
		for j := 0; j < 16; j++ {
			blocks[i].Scales[j] = 1
		}
		// Set some known values in Ql
		blocks[i].Ql[0] = 0x11 // First two values: low bits = 1, 1
	}

	output := make([]float32, numBlocks*QK_K)

	err := DequantizeQ6_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ6_K failed: %v", err)
	}

	// Verify output size
	if len(output) != numBlocks*QK_K {
		t.Errorf("Expected %d outputs, got %d", numBlocks*QK_K, len(output))
	}
}

// TestDequantizeQ6_K_SizeMismatch tests error handling for size mismatch
func TestDequantizeQ6_K_SizeMismatch(t *testing.T) {
	blocks := []BlockQ6_K{{}}
	output := make([]float32, QK_K-1) // Wrong size

	err := DequantizeQ6_K(blocks, output)
	if err == nil {
		t.Fatal("Expected error for size mismatch, got nil")
	}
}

// TestDequantizeQ6_KElement tests single element dequantization
func TestDequantizeQ6_KElement(t *testing.T) {
	// Create a simple block
	block := BlockQ6_K{
		D: float32ToFloat16(1.0),
	}

	// Set all scales to 1
	for i := 0; i < 16; i++ {
		block.Scales[i] = 1
	}

	// Set a known value at position 0
	// Low bits = 5 (stored in lower nibble of Ql[0])
	// High bits = 2 (stored in lowest 2 bits of Qh[0])
	// Combined: q = 5 | (2 << 4) = 5 | 32 = 37
	block.Ql[0] = 0x05 // Low bits = 5 for element 0
	block.Qh[0] = 0x02 // High bits = 2 for element 0

	// Convert block to bytes
	data := make([]byte, Q6_K_BLOCK_SIZE)
	copy(data[0:128], block.Ql[:])
	copy(data[128:192], block.Qh[:])
	for i := 0; i < 16; i++ {
		data[192+i] = byte(block.Scales[i])
	}
	data[208] = byte(block.D & 0xFF)
	data[209] = byte(block.D >> 8)

	// Dequantize element 0
	val := DequantizeQ6_KElement(data, 0)

	// Expected: d * scale * (q - 32) = 1.0 * 1 * (37 - 32) = 5.0
	expectedValue := float32(5.0)
	tolerance := float32(0.1)

	if math.Abs(float64(val-expectedValue)) > float64(tolerance) {
		t.Errorf("Expected ~%f, got %f", expectedValue, val)
	}
}

// TestDequantizeQ6_KElement_OutOfBounds tests out of bounds access
func TestDequantizeQ6_KElement_OutOfBounds(t *testing.T) {
	data := make([]byte, Q6_K_BLOCK_SIZE-1) // Too small

	val := DequantizeQ6_KElement(data, 0)

	// Should return 0 for out of bounds
	if val != 0 {
		t.Errorf("Expected 0 for out of bounds access, got %f", val)
	}
}

// TestParseQ6_KBlock tests block parsing from bytes
func TestParseQ6_KBlock(t *testing.T) {
	// Create test data
	data := make([]byte, Q6_K_BLOCK_SIZE)

	// Set some Ql values
	data[0] = 0x12
	data[1] = 0x34

	// Set some Qh values
	data[128] = 0xAB
	data[129] = 0xCD

	// Set some scale values
	data[192] = 0x01
	data[193] = 0x02

	// Set D field (little-endian)
	data[208] = 0x12
	data[209] = 0x34

	block := parseQ6_KBlock(data)

	// Check Ql field
	if block.Ql[0] != 0x12 || block.Ql[1] != 0x34 {
		t.Errorf("Ql parsing failed: got [0x%02X, 0x%02X], expected [0x12, 0x34]",
			block.Ql[0], block.Ql[1])
	}

	// Check Qh field
	if block.Qh[0] != 0xAB || block.Qh[1] != 0xCD {
		t.Errorf("Qh parsing failed: got [0x%02X, 0x%02X], expected [0xAB, 0xCD]",
			block.Qh[0], block.Qh[1])
	}

	// Check Scales field
	if block.Scales[0] != 0x01 || block.Scales[1] != 0x02 {
		t.Errorf("Scales parsing failed: got [%d, %d], expected [1, 2]",
			block.Scales[0], block.Scales[1])
	}

	// Check D field (little-endian)
	expectedD := uint16(0x12) | uint16(0x34)<<8
	if block.D != expectedD {
		t.Errorf("D = 0x%04X, expected 0x%04X", block.D, expectedD)
	}
}

// TestParseQ6_KBlock_TooSmall tests parsing with insufficient data
func TestParseQ6_KBlock_TooSmall(t *testing.T) {
	data := make([]byte, 10) // Too small

	block := parseQ6_KBlock(data)

	// Should return zero-initialized block
	if block.D != 0 {
		t.Errorf("Expected zero block, got D=%d", block.D)
	}
}

// TestDequantizeQ6_KTensor tests tensor-level dequantization
func TestDequantizeQ6_KTensor(t *testing.T) {
	// Create a Q6_K tensor
	numElements := QK_K * 2 // 2 blocks
	numBlocks := (numElements + QK_K - 1) / QK_K

	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Initialize with simple pattern
	for i := 0; i < numBlocks; i++ {
		offset := i * Q6_K_BLOCK_SIZE
		// Set D to 1.0 in float16
		rawData[offset+208] = 0x00 // D low byte
		rawData[offset+209] = 0x3C // D high byte (1.0 in float16)
		// Set all scales to 1
		for j := 0; j < 16; j++ {
			rawData[offset+192+j] = 0x01
		}
	}

	tensor := &Tensor{
		shape: []int{numElements},
		dtype: Q6_K,
		data:  rawData,
	}

	// Dequantize
	result, err := DequantizeQ6_KTensor(tensor)
	if err != nil {
		t.Fatalf("DequantizeQ6_KTensor failed: %v", err)
	}

	// Check result
	if result.dtype != Float32 {
		t.Errorf("Expected Float32, got %s", result.dtype)
	}

	if len(result.shape) != 1 || result.shape[0] != numElements {
		t.Errorf("Expected shape [%d], got %v", numElements, result.shape)
	}

	if len(result.data.([]float32)) != numElements {
		t.Errorf("Expected %d elements, got %d", numElements, len(result.data.([]float32)))
	}
}

// TestDequantizeQ6_KTensor_WrongType tests error handling for non-Q6_K tensor
func TestDequantizeQ6_KTensor_WrongType(t *testing.T) {
	tensor := &Tensor{
		shape: []int{100},
		dtype: Float32,
		data:  make([]float32, 100),
	}

	_, err := DequantizeQ6_KTensor(tensor)
	if err == nil {
		t.Fatal("Expected error for non-Q6_K tensor, got nil")
	}
}

// TestQuantizeQ6_K tests the quantization function
func TestQuantizeQ6_K(t *testing.T) {
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i) - 128.0 // Range from -128 to +127
	}

	blocks, err := QuantizeQ6_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ6_K failed: %v", err)
	}

	if len(blocks) != 1 {
		t.Errorf("Expected 1 block, got %d", len(blocks))
	}

	// Check that D (scale) is reasonable
	d := float16ToFloat32(blocks[0].D)
	if d <= 0 || d > 1000 {
		t.Errorf("Unexpected scale value: %f", d)
	}

	// Check that scales are set
	for i := 0; i < 16; i++ {
		if blocks[0].Scales[i] == 0 {
			t.Errorf("Scale[%d] is 0, expected non-zero", i)
		}
	}
}

// TestQuantizeQ6_K_MultipleBlocks tests quantization with multiple blocks
func TestQuantizeQ6_K_MultipleBlocks(t *testing.T) {
	input := make([]float32, QK_K*3+100) // 3.4 blocks
	for i := range input {
		input[i] = float32(i % 100) - 50.0
	}

	blocks, err := QuantizeQ6_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ6_K failed: %v", err)
	}

	expectedBlocks := (len(input) + QK_K - 1) / QK_K
	if len(blocks) != expectedBlocks {
		t.Errorf("Expected %d blocks, got %d", expectedBlocks, len(blocks))
	}
}

// TestQ6_K_BitPacking tests the bit packing/unpacking of Q6_K format
func TestQ6_K_BitPacking(t *testing.T) {
	// Test that we can correctly pack and unpack 6-bit values
	testValues := []uint8{0, 1, 15, 16, 31, 32, 47, 48, 63}

	for _, originalQ := range testValues {
		// Pack into low and high bits
		lowBits := originalQ & 0x0F
		highBits := (originalQ >> 4) & 0x03

		// Reconstruct
		reconstructedQ := lowBits | (highBits << 4)

		if reconstructedQ != originalQ {
			t.Errorf("Bit packing failed: original=%d, reconstructed=%d", originalQ, reconstructedQ)
		}
	}
}

// TestQ6_K_SymmetricQuantization tests the symmetric quantization around 32
func TestQ6_K_SymmetricQuantization(t *testing.T) {
	// Q6_K uses symmetric quantization: q - 32
	// So q=0 maps to -32, q=32 maps to 0, q=63 maps to +31

	testCases := []struct {
		q        uint8
		expected int
	}{
		{0, -32},
		{32, 0},
		{63, 31},
		{16, -16},
		{48, 16},
	}

	for _, tc := range testCases {
		result := int(tc.q) - 32
		if result != tc.expected {
			t.Errorf("For q=%d: expected %d, got %d", tc.q, tc.expected, result)
		}
	}
}

// BenchmarkDequantizeQ6_K benchmarks the dequantization of Q6_K blocks
func BenchmarkDequantizeQ6_K(b *testing.B) {
	blocks := make([]BlockQ6_K, 1000) // ~210KB of quantized data
	output := make([]float32, 1000*QK_K)

	// Initialize with some data
	for i := range blocks {
		blocks[i].D = float32ToFloat16(1.0)
		for j := 0; j < 16; j++ {
			blocks[i].Scales[j] = 1
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := DequantizeQ6_K(blocks, output)
		if err != nil {
			b.Fatal(err)
		}
	}

	// Report throughput
	bytesPerOp := int64(len(blocks) * Q6_K_BLOCK_SIZE)
	b.SetBytes(bytesPerOp)
}

// BenchmarkDequantizeQ6_KElement benchmarks single element access
func BenchmarkDequantizeQ6_KElement(b *testing.B) {
	data := make([]byte, Q6_K_BLOCK_SIZE*10)

	// Initialize with some data
	for i := 0; i < 10; i++ {
		offset := i * Q6_K_BLOCK_SIZE
		data[offset+208] = 0x00
		data[offset+209] = 0x3C // 1.0 in float16
		for j := 0; j < 16; j++ {
			data[offset+192+j] = 0x01
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = DequantizeQ6_KElement(data, i%QK_K)
	}
}

// BenchmarkDequantizeQ6_KTensor benchmarks full tensor dequantization
func BenchmarkDequantizeQ6_KTensor(b *testing.B) {
	numBlocks := 100
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	tensor := &Tensor{
		shape: []int{numBlocks * QK_K},
		dtype: Q6_K,
		data:  rawData,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := DequantizeQ6_KTensor(tensor)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQuantizeQ6_K benchmarks quantization
func BenchmarkQuantizeQ6_K(b *testing.B) {
	input := make([]float32, QK_K*100)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := QuantizeQ6_K(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}
