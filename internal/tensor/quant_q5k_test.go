package tensor

import (
	"math"
	"testing"
)

// TestDequantizeQ5_K_ZeroBlock tests dequantization of a zero-initialized block
func TestDequantizeQ5_K_ZeroBlock(t *testing.T) {
	blocks := []BlockQ5_K{{}}
	output := make([]float32, QK_K)

	err := DequantizeQ5_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ5_K failed: %v", err)
	}

	// All zeros should dequantize to zeros (or very small values)
	for i, val := range output {
		if math.Abs(float64(val)) > 0.01 {
			t.Errorf("Index %d: expected ~0, got %f", i, val)
		}
	}
}

// TestDequantizeQ5_K_KnownValues tests dequantization with known scale values
func TestDequantizeQ5_K_KnownValues(t *testing.T) {
	blocks := []BlockQ5_K{{
		D:    float32ToFloat16(1.0),    // scale = 1.0
		Dmin: float32ToFloat16(0.0),    // no offset
		Scales: [12]uint8{
			32, 170, 170, 10, 32, 170, 170, 10, 32, 170, 170, 10,
		},
		Qh: [32]uint8{0xFF, 0xFF, 0xFF, 0xFF}, // All high bits set
		Qs: [128]uint8{0xFF, 0xFF, 0xFF, 0xFF}, // All low bits set
	}}

	output := make([]float32, QK_K)

	err := DequantizeQ5_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ5_K failed: %v", err)
	}

	// With all bits set and scale=1, we expect high values
	// First value should be non-zero
	if output[0] == 0 {
		t.Errorf("Expected non-zero output for first value, got 0")
	}
}

// TestDequantizeQ5_K_Roundtrip tests quantization followed by dequantization
func TestDequantizeQ5_K_Roundtrip(t *testing.T) {
	// Create test data with known pattern
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i % 32) // Pattern 0-31 repeating
	}

	// Quantize
	blocks, err := QuantizeQ5_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ5_K failed: %v", err)
	}

	// Dequantize
	output := make([]float32, QK_K)
	err = DequantizeQ5_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ5_K failed: %v", err)
	}

	// Check that values are approximately correct (within quantization error)
	const tolerance = 2.0 // Q5_K has 5 bits, so max error is ~range/32
	for i := range input {
		diff := math.Abs(float64(input[i] - output[i]))
		if diff > tolerance {
			t.Errorf("Index %d: input=%f, output=%f, diff=%f (tolerance=%f)",
				i, input[i], output[i], diff, tolerance)
		}
	}
}

// TestDequantizeQ5_K_MultipleBlocks tests with multiple blocks
func TestDequantizeQ5_K_MultipleBlocks(t *testing.T) {
	numBlocks := 4
	blocks := make([]BlockQ5_K, numBlocks)

	// Initialize each block with different scale
	for i := range blocks {
		blocks[i].D = float32ToFloat16(float32(i + 1))
		blocks[i].Dmin = float32ToFloat16(0.0)
		// Set some known values in Qs
		blocks[i].Qs[0] = 0x11 // First two values: 1, 1
	}

	output := make([]float32, numBlocks*QK_K)

	err := DequantizeQ5_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ5_K failed: %v", err)
	}

	// Verify output size
	if len(output) != numBlocks*QK_K {
		t.Errorf("Expected %d outputs, got %d", numBlocks*QK_K, len(output))
	}
}

// TestDequantizeQ5_K_SizeMismatch tests error handling for size mismatch
func TestDequantizeQ5_K_SizeMismatch(t *testing.T) {
	blocks := []BlockQ5_K{{}}
	output := make([]float32, QK_K-1) // Wrong size

	err := DequantizeQ5_K(blocks, output)
	if err == nil {
		t.Fatal("Expected error for size mismatch, got nil")
	}
}

// TestExtractScalesAndMins tests the scale/min extraction function
func TestExtractScalesAndMins(t *testing.T) {
	// Test with known bit pattern
	scales := [12]uint8{
		0x00, 0x00, 0x00, // Scales: all zeros
		0x00, 0x00, 0x00,
		0xFF, 0xFF, 0xFF, // Mins: all ones
		0xFF, 0xFF, 0xFF,
	}

	sc, m := extractScalesAndMins(scales)

	// Check that scales are extracted (should be 0 or small values)
	for i, val := range sc {
		if val > 63 {
			t.Errorf("Scale[%d] = %d, expected 0-63", i, val)
		}
	}

	// Check that mins are extracted (should be non-zero)
	for i, val := range m {
		if val == 0 {
			t.Errorf("Min[%d] = 0, expected non-zero from 0xFF bytes", i)
		}
		if val > 63 {
			t.Errorf("Min[%d] = %d, expected 0-63", i, val)
		}
	}
}

// TestExtractScalesAndMins_AllOnes tests extraction with all bits set
func TestExtractScalesAndMins_AllOnes(t *testing.T) {
	scales := [12]uint8{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}

	sc, m := extractScalesAndMins(scales)

	// All 6-bit values should be 63 (0x3F)
	for i, val := range sc {
		if val != 63 {
			t.Errorf("Scale[%d] = %d, expected 63", i, val)
		}
	}

	for i, val := range m {
		if val != 63 {
			t.Errorf("Min[%d] = %d, expected 63", i, val)
		}
	}
}

// TestDequantizeQ5_KElement tests single element dequantization
func TestDequantizeQ5_KElement(t *testing.T) {
	// Create a simple block
	block := BlockQ5_K{
		D:    float32ToFloat16(1.0),
		Dmin: float32ToFloat16(0.0),
		Scales: [12]uint8{
			32, 170, 170, 10, 32, 170, 170, 10, 32, 170, 170, 10,
		},
	}

	// Set a known value at position 0
	block.Qs[0] = 0x05 // Low bits = 5 for element 0
	block.Qh[0] = 0x00 // High bit = 0 for element 0

	// Convert block to bytes
	data := make([]byte, Q5_K_BLOCK_SIZE)
	data[0] = byte(block.D & 0xFF)
	data[1] = byte(block.D >> 8)
	data[2] = byte(block.Dmin & 0xFF)
	data[3] = byte(block.Dmin >> 8)
	copy(data[4:16], block.Scales[:])
	copy(data[16:48], block.Qh[:])
	copy(data[48:176], block.Qs[:])

	// Dequantize element 0
	val := DequantizeQ5_KElement(data, 0)

	// Should get a non-zero value based on quantized value 5
	if val == 0 {
		t.Errorf("Expected non-zero value, got 0")
	}
}

// TestDequantizeQ5_KElement_OutOfBounds tests out of bounds access
func TestDequantizeQ5_KElement_OutOfBounds(t *testing.T) {
	data := make([]byte, Q5_K_BLOCK_SIZE-1) // Too small

	val := DequantizeQ5_KElement(data, 0)

	// Should return 0 for out of bounds
	if val != 0 {
		t.Errorf("Expected 0 for out of bounds access, got %f", val)
	}
}

// TestParseQ5_KBlock tests block parsing from bytes
func TestParseQ5_KBlock(t *testing.T) {
	// Create test data
	data := make([]byte, Q5_K_BLOCK_SIZE)
	data[0] = 0x12
	data[1] = 0x34
	data[2] = 0x56
	data[3] = 0x78

	block := parseQ5_KBlock(data)

	// Check D field (little-endian)
	expectedD := uint16(0x12) | uint16(0x34)<<8
	if block.D != expectedD {
		t.Errorf("D = 0x%04X, expected 0x%04X", block.D, expectedD)
	}

	// Check Dmin field (little-endian)
	expectedDmin := uint16(0x56) | uint16(0x78)<<8
	if block.Dmin != expectedDmin {
		t.Errorf("Dmin = 0x%04X, expected 0x%04X", block.Dmin, expectedDmin)
	}
}

// TestParseQ5_KBlock_TooSmall tests parsing with insufficient data
func TestParseQ5_KBlock_TooSmall(t *testing.T) {
	data := make([]byte, 10) // Too small

	block := parseQ5_KBlock(data)

	// Should return zero-initialized block
	if block.D != 0 || block.Dmin != 0 {
		t.Errorf("Expected zero block, got D=%d, Dmin=%d", block.D, block.Dmin)
	}
}

// TestDequantizeQ5_KTensor tests tensor-level dequantization
func TestDequantizeQ5_KTensor(t *testing.T) {
	// Create a Q5_K tensor
	numElements := QK_K * 2 // 2 blocks
	numBlocks := (numElements + QK_K - 1) / QK_K

	rawData := make([]byte, numBlocks*Q5_K_BLOCK_SIZE)

	// Initialize with simple pattern
	for i := 0; i < numBlocks; i++ {
		offset := i * Q5_K_BLOCK_SIZE
		rawData[offset] = 0x00   // D low byte
		rawData[offset+1] = 0x3C // D high byte (1.0 in float16)
		rawData[offset+2] = 0x00 // Dmin
		rawData[offset+3] = 0x00
	}

	tensor := &Tensor{
		shape: []int{numElements},
		dtype: Q5_K,
		data:  rawData,
	}

	// Dequantize
	result, err := DequantizeQ5_KTensor(tensor)
	if err != nil {
		t.Fatalf("DequantizeQ5_KTensor failed: %v", err)
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

// TestDequantizeQ5_KTensor_WrongType tests error handling for non-Q5_K tensor
func TestDequantizeQ5_KTensor_WrongType(t *testing.T) {
	tensor := &Tensor{
		shape: []int{100},
		dtype: Float32,
		data:  make([]float32, 100),
	}

	_, err := DequantizeQ5_KTensor(tensor)
	if err == nil {
		t.Fatal("Expected error for non-Q5_K tensor, got nil")
	}
}

// TestFloat32ToFloat16 tests float32 to float16 conversion
func TestFloat32ToFloat16(t *testing.T) {
	tests := []struct {
		input    float32
		expected uint16
	}{
		{0.0, 0x0000},
		{1.0, 0x3C00},
		{-1.0, 0xBC00},
	}

	for _, tt := range tests {
		result := float32ToFloat16(tt.input)
		if result != tt.expected {
			t.Errorf("float32ToFloat16(%f) = 0x%04X, expected 0x%04X",
				tt.input, result, tt.expected)
		}
	}
}

// TestFloat32ToFloat16_Roundtrip tests conversion roundtrip
func TestFloat32ToFloat16_Roundtrip(t *testing.T) {
	testValues := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0}

	for _, val := range testValues {
		f16 := float32ToFloat16(val)
		f32 := float16ToFloat32(f16)

		// Check that roundtrip is approximately correct
		diff := math.Abs(float64(val - f32))
		tolerance := math.Abs(float64(val)) * 0.001 // 0.1% tolerance

		if diff > tolerance && diff > 0.001 {
			t.Errorf("Roundtrip failed for %f: got %f (diff=%f, tolerance=%f)",
				val, f32, diff, tolerance)
		}
	}
}

// TestQuantizeQ5_K tests the quantization function
func TestQuantizeQ5_K(t *testing.T) {
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i)
	}

	blocks, err := QuantizeQ5_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ5_K failed: %v", err)
	}

	if len(blocks) != 1 {
		t.Errorf("Expected 1 block, got %d", len(blocks))
	}

	// Check that D (scale) is reasonable
	d := float16ToFloat32(blocks[0].D)
	if d <= 0 || d > 1000 {
		t.Errorf("Unexpected scale value: %f", d)
	}
}

// TestQuantizeQ5_K_MultipleBlocks tests quantization with multiple blocks
func TestQuantizeQ5_K_MultipleBlocks(t *testing.T) {
	input := make([]float32, QK_K*3+100) // 3.4 blocks
	for i := range input {
		input[i] = float32(i % 100)
	}

	blocks, err := QuantizeQ5_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ5_K failed: %v", err)
	}

	expectedBlocks := (len(input) + QK_K - 1) / QK_K
	if len(blocks) != expectedBlocks {
		t.Errorf("Expected %d blocks, got %d", expectedBlocks, len(blocks))
	}
}

// BenchmarkDequantizeQ5_K benchmarks the dequantization of Q5_K blocks
func BenchmarkDequantizeQ5_K(b *testing.B) {
	blocks := make([]BlockQ5_K, 1000) // ~176KB of quantized data
	output := make([]float32, 1000*QK_K)

	// Initialize with some data
	for i := range blocks {
		blocks[i].D = float32ToFloat16(1.0)
		blocks[i].Dmin = float32ToFloat16(0.0)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := DequantizeQ5_K(blocks, output)
		if err != nil {
			b.Fatal(err)
		}
	}

	// Report throughput
	bytesPerOp := int64(len(blocks) * Q5_K_BLOCK_SIZE)
	b.SetBytes(bytesPerOp)
}

// BenchmarkDequantizeQ5_KElement benchmarks single element access
func BenchmarkDequantizeQ5_KElement(b *testing.B) {
	data := make([]byte, Q5_K_BLOCK_SIZE*10)

	// Initialize with some data
	for i := 0; i < 10; i++ {
		offset := i * Q5_K_BLOCK_SIZE
		data[offset] = 0x00
		data[offset+1] = 0x3C // 1.0 in float16
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = DequantizeQ5_KElement(data, i%QK_K)
	}
}

// BenchmarkDequantizeQ5_KTensor benchmarks full tensor dequantization
func BenchmarkDequantizeQ5_KTensor(b *testing.B) {
	numBlocks := 100
	rawData := make([]byte, numBlocks*Q5_K_BLOCK_SIZE)

	tensor := &Tensor{
		shape: []int{numBlocks * QK_K},
		dtype: Q5_K,
		data:  rawData,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := DequantizeQ5_KTensor(tensor)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQuantizeQ5_K benchmarks quantization
func BenchmarkQuantizeQ5_K(b *testing.B) {
	input := make([]float32, QK_K*100)
	for i := range input {
		input[i] = float32(i)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := QuantizeQ5_K(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}
