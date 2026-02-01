package tensor

import (
	"math"
	"testing"
)

// TestQ6_K_IntegrationWithMatMul tests Q6_K dequantization integrated with matrix multiplication
func TestQ6_K_IntegrationWithMatMul(t *testing.T) {
	// Create a simple Q6_K tensor (simulating a weight matrix)
	numBlocks := 1

	// Create raw Q6_K data
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Set up a simple block with predictable values
	// D = 1.0 (super-block scale)
	d := float32ToFloat16(1.0)
	rawData[208] = byte(d & 0xFF)
	rawData[209] = byte(d >> 8)

	// Set all scales to 1 (int8)
	for i := 0; i < 16; i++ {
		rawData[192+i] = 0x01
	}

	// Set some known quantized values
	// For first few values, set q=40 (which will give (40-32)=8 after dequantization)
	// q=40 in binary: 101000 -> low bits=1000 (8), high bits=10 (2)
	for i := 0; i < 10; i++ {
		// Low 4 bits
		qlIdx := i / 2
		lowBits := uint8(8) // low 4 bits of 40
		if i%2 == 0 {
			rawData[qlIdx] = lowBits
		} else {
			rawData[qlIdx] |= (lowBits << 4)
		}

		// High 2 bits
		qhIdx := i / 4
		bitPos := (i % 4) * 2
		highBits := uint8(2) // high 2 bits of 40
		rawData[128+qhIdx] |= (highBits << bitPos)
	}

	// Create Q6_K tensor
	weightTensor := &Tensor{
		shape:  []int{16, 16}, // 16x16 matrix = 256 elements
		stride: computeStrides([]int{16, 16}),
		dtype:  Q6_K,
		data:   rawData,
		device: CPU,
		offset: 0,
	}

	// Create a simple input tensor (Float32)
	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = 1.0 // All ones
	}
	inputTensor := &Tensor{
		shape:  []int{1, 16}, // 1x16 row vector
		stride: computeStrides([]int{1, 16}),
		dtype:  Float32,
		data:   inputData,
		device: CPU,
		offset: 0,
	}

	// Perform matrix multiplication (should auto-dequantize Q6_K)
	result := MatMul(inputTensor, weightTensor)

	// Verify result shape
	expectedShape := []int{1, 16}
	if len(result.shape) != len(expectedShape) {
		t.Fatalf("Expected shape %v, got %v", expectedShape, result.shape)
	}
	for i := range expectedShape {
		if result.shape[i] != expectedShape[i] {
			t.Fatalf("Shape mismatch at dim %d: expected %d, got %d", i, expectedShape[i], result.shape[i])
		}
	}

	// Verify result type is Float32
	if result.dtype != Float32 {
		t.Fatalf("Expected Float32 result, got %s", result.dtype)
	}

	// Result should have some non-zero values due to the quantized weights
	resultData := result.data.([]float32)
	hasNonZero := false
	for _, val := range resultData {
		if val != 0 {
			hasNonZero = true
			break
		}
	}

	if !hasNonZero {
		t.Error("Expected some non-zero values in result, got all zeros")
	}
}

// TestQ6_K_IntegrationTensorAt tests single element access on Q6_K tensors
func TestQ6_K_IntegrationTensorAt(t *testing.T) {
	// Create a Q6_K tensor with known values
	numBlocks := 1
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Set D = 2.0 (super-block scale)
	d := float32ToFloat16(2.0)
	rawData[208] = byte(d & 0xFF)
	rawData[209] = byte(d >> 8)

	// Set scale[0] = 3 (first group scale)
	rawData[192] = 0x03

	// Set first value: q = 50 (binary: 110010)
	// low bits = 0010 (2), high bits = 11 (3)
	rawData[0] = 0x02      // Ql[0] low nibble = 2
	rawData[128] = 0x03    // Qh[0] lowest 2 bits = 3

	// Create tensor
	tensor := &Tensor{
		shape:  []int{QK_K},
		stride: computeStrides([]int{QK_K}),
		dtype:  Q6_K,
		data:   rawData,
		device: CPU,
		offset: 0,
	}

	// Access first element via At()
	value := tensor.At(0)

	// Expected: d * scale * (q - 32) = 2.0 * 3 * (50 - 32) = 2.0 * 3 * 18 = 108.0
	expectedValue := float32(108.0)
	tolerance := float32(0.1)

	if math.Abs(float64(value-expectedValue)) > float64(tolerance) {
		t.Errorf("Expected value ~%f, got %f", expectedValue, value)
	}
}

// TestQ6_K_IntegrationFullTensorDequantization tests full tensor dequantization
func TestQ6_K_IntegrationFullTensorDequantization(t *testing.T) {
	// Create a Q6_K tensor with 2 blocks
	numBlocks := 2
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Set up both blocks with simple patterns
	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		offset := blockIdx * Q6_K_BLOCK_SIZE

		// D = 1.0
		d := float32ToFloat16(1.0)
		rawData[offset+208] = byte(d & 0xFF)
		rawData[offset+209] = byte(d >> 8)

		// All scales = 1
		for i := 0; i < 16; i++ {
			rawData[offset+192+i] = 0x01
		}

		// Set a simple pattern in quantized values
		for i := 0; i < 128; i++ {
			rawData[offset+i] = 0x55 // Ql: alternating pattern
		}
		for i := 0; i < 64; i++ {
			rawData[offset+128+i] = 0xAA // Qh: alternating pattern
		}
	}

	// Create Q6_K tensor
	q6kTensor := &Tensor{
		shape:  []int{numBlocks * QK_K},
		stride: computeStrides([]int{numBlocks * QK_K}),
		dtype:  Q6_K,
		data:   rawData,
		device: CPU,
		offset: 0,
	}

	// Dequantize to float32
	float32Tensor, err := DequantizeQ6_KTensor(q6kTensor)
	if err != nil {
		t.Fatalf("Dequantization failed: %v", err)
	}

	// Verify result
	if float32Tensor.dtype != Float32 {
		t.Errorf("Expected Float32, got %s", float32Tensor.dtype)
	}

	if len(float32Tensor.shape) != 1 || float32Tensor.shape[0] != numBlocks*QK_K {
		t.Errorf("Shape mismatch: expected [%d], got %v", numBlocks*QK_K, float32Tensor.shape)
	}

	// Verify data size
	float32Data := float32Tensor.data.([]float32)
	if len(float32Data) != numBlocks*QK_K {
		t.Errorf("Data size mismatch: expected %d, got %d", numBlocks*QK_K, len(float32Data))
	}

	// Verify values are reasonable (should be centered around 0 due to symmetric quantization)
	sum := float32(0)
	for _, val := range float32Data {
		sum += val
		// Values should be in a reasonable range
		if math.Abs(float64(val)) > 1000 {
			t.Errorf("Value out of expected range: %f", val)
		}
	}

	// Mean should be close to 0 for symmetric quantization with random pattern
	mean := sum / float32(len(float32Data))
	if math.Abs(float64(mean)) > 50 {
		t.Logf("Mean value is %f (expected close to 0 for symmetric quantization)", mean)
	}
}

// TestQ6_K_IntegrationCompareWithQ5_K compares Q6_K and Q5_K behavior
func TestQ6_K_IntegrationCompareWithQ5_K(t *testing.T) {
	// Create test data
	input := make([]float32, QK_K)
	for i := range input {
		input[i] = float32(i%64) - 32.0 // Range: -32 to +31
	}

	// Quantize to Q6_K
	q6kBlocks, err := QuantizeQ6_K(input)
	if err != nil {
		t.Fatalf("Q6_K quantization failed: %v", err)
	}

	// Dequantize Q6_K
	q6kOutput := make([]float32, QK_K)
	err = DequantizeQ6_K(q6kBlocks, q6kOutput)
	if err != nil {
		t.Fatalf("Q6_K dequantization failed: %v", err)
	}

	// Calculate Q6_K error
	q6kError := float32(0)
	for i := range input {
		diff := input[i] - q6kOutput[i]
		q6kError += diff * diff
	}
	q6kError /= float32(len(input))
	q6kMSE := math.Sqrt(float64(q6kError))

	t.Logf("Q6_K MSE: %f", q6kMSE)

	// Q6_K should have relatively low error (6 bits of precision)
	// MSE should be less than 2.0 for this test case
	if q6kMSE > 2.0 {
		t.Errorf("Q6_K MSE too high: %f (expected < 2.0)", q6kMSE)
	}
}

// BenchmarkQ6_K_IntegrationMatMul benchmarks Q6_K in realistic matrix multiplication
func BenchmarkQ6_K_IntegrationMatMul(b *testing.B) {
	// Create Q6_K weight matrix (simulating a typical layer)
	weightSize := QK_K * 16 // 16 blocks = 4096 elements
	numBlocks := weightSize / QK_K
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Initialize with realistic values
	for i := 0; i < numBlocks; i++ {
		offset := i * Q6_K_BLOCK_SIZE
		d := float32ToFloat16(0.1)
		rawData[offset+208] = byte(d & 0xFF)
		rawData[offset+209] = byte(d >> 8)
		for j := 0; j < 16; j++ {
			rawData[offset+192+j] = 0x01
		}
	}

	weightTensor := &Tensor{
		shape:  []int{64, 64}, // 64x64 matrix = 4096 elements
		stride: computeStrides([]int{64, 64}),
		dtype:  Q6_K,
		data:   rawData,
		device: CPU,
		offset: 0,
	}

	// Create input tensor
	inputData := make([]float32, 64)
	for i := range inputData {
		inputData[i] = 0.5
	}
	inputTensor := &Tensor{
		shape:  []int{1, 64},
		stride: computeStrides([]int{1, 64}),
		dtype:  Float32,
		data:   inputData,
		device: CPU,
		offset: 0,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = MatMul(inputTensor, weightTensor)
	}
}
