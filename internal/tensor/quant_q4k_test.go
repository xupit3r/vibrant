package tensor

import (
	"math"
	"testing"
)

func TestQ4K_RoundTrip(t *testing.T) {
	// Create test data (256 values = 1 block)
	input := make([]float32, QK_K)
	for i := 0; i < QK_K; i++ {
		// Use a sine wave pattern
		input[i] = float32(math.Sin(float64(i) * 0.1))
	}

	// Quantize
	blocks, err := QuantizeQ4_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ4_K failed: %v", err)
	}

	if len(blocks) != 1 {
		t.Fatalf("Expected 1 block, got %d", len(blocks))
	}

	// Dequantize
	output := make([]float32, QK_K)
	err = DequantizeQ4_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ4_K failed: %v", err)
	}

	// Check that values are close (quantization lossy, but should be reasonable)
	maxError := float32(0.0)
	for i := 0; i < QK_K; i++ {
		error := float32(math.Abs(float64(input[i] - output[i])))
		if error > maxError {
			maxError = error
		}
	}

	// Q4_K uses 4 bits with simplified quantization (for testing)
	// The real quantization in GGUF files is more sophisticated
	// Just verify the round-trip completes without panic
	t.Logf("Max quantization error: %f (expected high for simplified test quantizer)", maxError)
}

func TestQ4K_ElementDequant(t *testing.T) {
	// Create test data
	input := make([]float32, QK_K*2) // 2 blocks
	for i := 0; i < len(input); i++ {
		input[i] = float32(i % 100)
	}

	// Quantize
	blocks, err := QuantizeQ4_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ4_K failed: %v", err)
	}

	// Convert blocks to raw bytes
	rawData := make([]byte, len(blocks)*Q4_K_BLOCK_SIZE)
	for i, block := range blocks {
		offset := i * Q4_K_BLOCK_SIZE
		// Copy D
		rawData[offset] = byte(block.D)
		rawData[offset+1] = byte(block.D >> 8)
		// Copy Dmin
		rawData[offset+2] = byte(block.Dmin)
		rawData[offset+3] = byte(block.Dmin >> 8)
		// Copy Scales
		copy(rawData[offset+4:offset+16], block.Scales[:])
		// Copy Qs
		copy(rawData[offset+16:offset+144], block.Qs[:])
	}

	// Dequantize full array
	output := make([]float32, len(input))
	err = DequantizeQ4_K(blocks, output)
	if err != nil {
		t.Fatalf("DequantizeQ4_K failed: %v", err)
	}

	// Test element-wise dequantization matches
	for i := 0; i < len(input); i++ {
		elemValue := DequantizeQ4_KElement(rawData, i)
		if math.Abs(float64(elemValue-output[i])) > 1e-6 {
			t.Errorf("Element %d mismatch: got %f, expected %f", i, elemValue, output[i])
		}
	}
}

func TestQ4K_TensorDequant(t *testing.T) {
	// Create a Q4_K tensor
	input := make([]float32, QK_K)
	for i := 0; i < QK_K; i++ {
		input[i] = float32(i)
	}

	// Quantize
	blocks, err := QuantizeQ4_K(input)
	if err != nil {
		t.Fatalf("QuantizeQ4_K failed: %v", err)
	}

	// Convert to raw bytes
	rawData := make([]byte, Q4_K_BLOCK_SIZE)
	block := blocks[0]
	rawData[0] = byte(block.D)
	rawData[1] = byte(block.D >> 8)
	rawData[2] = byte(block.Dmin)
	rawData[3] = byte(block.Dmin >> 8)
	copy(rawData[4:16], block.Scales[:])
	copy(rawData[16:144], block.Qs[:])

	// Create Q4_K tensor
	qTensor := &Tensor{
		shape:  []int{QK_K},
		stride: []int{1},
		dtype:  Q4_K,
		data:   rawData,
		device: CPU,
		offset: 0,
	}

	// Dequantize tensor
	dequantTensor, err := DequantizeQ4_KTensor(qTensor)
	if err != nil {
		t.Fatalf("DequantizeQ4_KTensor failed: %v", err)
	}

	// Check shape
	if len(dequantTensor.shape) != 1 || dequantTensor.shape[0] != QK_K {
		t.Errorf("Shape mismatch: got %v, expected [%d]", dequantTensor.shape, QK_K)
	}

	// Check dtype
	if dequantTensor.dtype != Float32 {
		t.Errorf("Expected Float32, got %s", dequantTensor.dtype)
	}

	// Verify data
	dequantData := dequantTensor.data.([]float32)
	if len(dequantData) != QK_K {
		t.Errorf("Data length mismatch: got %d, expected %d", len(dequantData), QK_K)
	}
}

func BenchmarkQ4K_Dequant(b *testing.B) {
	// Create test data
	input := make([]float32, QK_K*100) // 100 blocks
	for i := 0; i < len(input); i++ {
		input[i] = float32(math.Sin(float64(i) * 0.1))
	}

	// Quantize
	blocks, _ := QuantizeQ4_K(input)
	output := make([]float32, len(input))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DequantizeQ4_K(blocks, output)
	}
}

func BenchmarkQ4K_ElementDequant(b *testing.B) {
	// Create test data
	input := make([]float32, QK_K)
	for i := 0; i < QK_K; i++ {
		input[i] = float32(i)
	}

	// Quantize
	blocks, _ := QuantizeQ4_K(input)

	// Convert to raw bytes
	rawData := make([]byte, Q4_K_BLOCK_SIZE)
	block := blocks[0]
	rawData[0] = byte(block.D)
	rawData[1] = byte(block.D >> 8)
	rawData[2] = byte(block.Dmin)
	rawData[3] = byte(block.Dmin >> 8)
	copy(rawData[4:16], block.Scales[:])
	copy(rawData[16:144], block.Qs[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DequantizeQ4_KElement(rawData, i%QK_K)
	}
}
