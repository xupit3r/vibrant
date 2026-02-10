package tensor

import (
	"math"
	"testing"
)

// buildQ6KTestTensor creates a small Q6_K tensor for testing.
// Dimensions must be multiples of QK_K (256).
func buildQ6KTestTensor(M, N int) *Tensor {
	totalElements := M * N
	numBlocks := totalElements / QK_K
	rawData := make([]byte, numBlocks*Q6_K_BLOCK_SIZE)

	// Fill with deterministic test data
	for i := 0; i < numBlocks; i++ {
		off := i * Q6_K_BLOCK_SIZE
		// Set a non-zero D scale (float16 encoded)
		rawData[off+208] = 0x00 // D low byte
		rawData[off+209] = 0x3C // D high byte (~1.0 in float16)
		// Set some scale values
		for j := 0; j < 16; j++ {
			rawData[off+128+64+j] = byte(j + 1) // Scales[j]
		}
		// Set some quantized values
		for j := 0; j < 128; j++ {
			rawData[off+j] = byte((j * 3) & 0xFF)
		}
	}

	return &Tensor{
		data:   rawData,
		shape:  []int{M, N},
		stride: computeStrides([]int{M, N}),
		dtype:  Q6_K,
		device: CPU,
	}
}

// buildQ4KTestTensor creates a small Q4_K tensor for testing.
func buildQ4KTestTensor(M, N int) *Tensor {
	totalElements := M * N
	numBlocks := totalElements / QK_K
	rawData := make([]byte, numBlocks*Q4_K_BLOCK_SIZE)

	for i := 0; i < numBlocks; i++ {
		off := i * Q4_K_BLOCK_SIZE
		// D
		rawData[off+0] = 0x00
		rawData[off+1] = 0x3C
		// Dmin
		rawData[off+2] = 0x00
		rawData[off+3] = 0x38
		// Scales
		for j := 0; j < 12; j++ {
			rawData[off+4+j] = byte(j + 1)
		}
		// Qs
		for j := 0; j < 128; j++ {
			rawData[off+16+j] = byte((j * 5) & 0xFF)
		}
	}

	return &Tensor{
		data:   rawData,
		shape:  []int{M, N},
		stride: computeStrides([]int{M, N}),
		dtype:  Q4_K,
		device: CPU,
	}
}

// buildQ5KTestTensor creates a small Q5_K tensor for testing.
func buildQ5KTestTensor(M, N int) *Tensor {
	totalElements := M * N
	numBlocks := totalElements / QK_K
	rawData := make([]byte, numBlocks*Q5_K_BLOCK_SIZE)

	for i := 0; i < numBlocks; i++ {
		off := i * Q5_K_BLOCK_SIZE
		// D
		rawData[off+0] = 0x00
		rawData[off+1] = 0x3C
		// Dmin
		rawData[off+2] = 0x00
		rawData[off+3] = 0x38
		// Scales (12 bytes)
		for j := 0; j < 12; j++ {
			rawData[off+4+j] = byte(j + 1)
		}
		// Qh (32 bytes)
		for j := 0; j < 32; j++ {
			rawData[off+16+j] = byte((j * 7) & 0xFF)
		}
		// Qs (128 bytes)
		for j := 0; j < 128; j++ {
			rawData[off+48+j] = byte((j * 3) & 0xFF)
		}
	}

	return &Tensor{
		data:   rawData,
		shape:  []int{M, N},
		stride: computeStrides([]int{M, N}),
		dtype:  Q5_K,
		device: CPU,
	}
}

// verifyTransposeCorrectness checks that fused matches separate dequant+transpose.
func verifyTransposeCorrectness(t *testing.T, name string, fused *Tensor, original *Tensor) {
	// Separate path: dequant then transpose
	dequant := Dequantize(original)
	transposed := dequant.Transpose()

	fusedData := fused.data.([]float32)
	transposedData := transposed.data.([]float32)

	if len(fusedData) != len(transposedData) {
		t.Fatalf("%s: length mismatch: fused=%d, separate=%d", name, len(fusedData), len(transposedData))
	}

	maxDiff := float32(0)
	for i := range fusedData {
		diff := float32(math.Abs(float64(fusedData[i] - transposedData[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 1e-6 {
		t.Errorf("%s: max difference %.8f exceeds tolerance", name, maxDiff)
	}

	// Verify shape is transposed
	M, N := original.shape[0], original.shape[1]
	if fused.shape[0] != N || fused.shape[1] != M {
		t.Errorf("%s: expected shape [%d, %d], got %v", name, N, M, fused.shape)
	}

	if !fused.transposed {
		t.Errorf("%s: fused result should be marked as transposed", name)
	}
}

func TestDequantTransposeQ6K(t *testing.T) {
	// 256x256 = 65536 elements = 256 blocks
	tensor := buildQ6KTestTensor(256, 256)
	fused, err := DequantTransposeQ6K(tensor)
	if err != nil {
		t.Fatalf("DequantTransposeQ6K failed: %v", err)
	}
	verifyTransposeCorrectness(t, "Q6_K", fused, tensor)
}

func TestDequantTransposeQ4K(t *testing.T) {
	tensor := buildQ4KTestTensor(256, 256)
	fused, err := DequantTransposeQ4K(tensor)
	if err != nil {
		t.Fatalf("DequantTransposeQ4K failed: %v", err)
	}
	verifyTransposeCorrectness(t, "Q4_K", fused, tensor)
}

func TestDequantTransposeQ5K(t *testing.T) {
	tensor := buildQ5KTestTensor(256, 256)
	fused, err := DequantTransposeQ5K(tensor)
	if err != nil {
		t.Fatalf("DequantTransposeQ5K failed: %v", err)
	}
	verifyTransposeCorrectness(t, "Q5_K", fused, tensor)
}

func TestDequantTransposeWrongDtype(t *testing.T) {
	tensor := NewTensor([]int{256, 256}, Float32)

	_, err := DequantTransposeQ6K(tensor)
	if err == nil {
		t.Error("expected error for non-Q6_K tensor")
	}

	_, err = DequantTransposeQ4K(tensor)
	if err == nil {
		t.Error("expected error for non-Q4_K tensor")
	}

	_, err = DequantTransposeQ5K(tensor)
	if err == nil {
		t.Error("expected error for non-Q5_K tensor")
	}
}

func TestDequantTransposeNon2D(t *testing.T) {
	rawData := make([]byte, Q6_K_BLOCK_SIZE)
	tensor := &Tensor{
		data:   rawData,
		shape:  []int{256},
		stride: computeStrides([]int{256}),
		dtype:  Q6_K,
		device: CPU,
	}

	_, err := DequantTransposeQ6K(tensor)
	if err == nil {
		t.Error("expected error for 1D tensor")
	}
}

func BenchmarkDequantTransposeQ6K_Fused(b *testing.B) {
	tensor := buildQ6KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DequantTransposeQ6K(tensor)
	}
}

func BenchmarkDequantTransposeQ6K_Separate(b *testing.B) {
	tensor := buildQ6KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequant := Dequantize(tensor)
		_ = dequant.Transpose()
	}
}

func BenchmarkDequantTransposeQ4K_Fused(b *testing.B) {
	tensor := buildQ4KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DequantTransposeQ4K(tensor)
	}
}

func BenchmarkDequantTransposeQ4K_Separate(b *testing.B) {
	tensor := buildQ4KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequant := Dequantize(tensor)
		_ = dequant.Transpose()
	}
}

func BenchmarkDequantTransposeQ5K_Fused(b *testing.B) {
	tensor := buildQ5KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DequantTransposeQ5K(tensor)
	}
}

func BenchmarkDequantTransposeQ5K_Separate(b *testing.B) {
	tensor := buildQ5KTestTensor(768, 768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequant := Dequantize(tensor)
		_ = dequant.Transpose()
	}
}
