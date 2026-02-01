package tensor

import (
	"fmt"
	"math"
	"testing"
)

// ============================================================================
// Correctness Tests (vs Reference Implementation)
// ============================================================================

// TestMatMulQ5K_Correctness validates that fused dequant+matmul produces
// identical results to the current approach (dequant then matmul).
func TestMatMulQ5K_Correctness(t *testing.T) {
	// Create small test matrices for easy verification
	M, K, N := 4, 8, 4

	// Create input matrix A (Float32)
	a := NewTensor([]int{M, K}, Float32)
	aData := a.data.([]float32)
	for i := 0; i < M*K; i++ {
		aData[i] = float32(i%10) * 0.1 // Values 0.0, 0.1, 0.2, ..., 0.9
	}

	// Create weight matrix B (Float32, will be quantized)
	bFloat := NewTensor([]int{K, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := 0; i < K*N; i++ {
		bFloatData[i] = float32(i%20) * 0.5 // Values 0.0, 0.5, 1.0, ..., 9.5
	}

	// Quantize B to Q5_K
	bQuant, err := QuantizeQ5_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize tensor: %v", err)
	}

	// Method 1: Current approach (dequant then matmul)
	bDequant, err := DequantizeQ5_KTensor(bQuant)
	if err != nil {
		t.Fatalf("Failed to dequantize tensor: %v", err)
	}
	expected := MatMul(a, bDequant)

	// Method 2: Fused approach
	result, err := MatMulQ5K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ5K failed: %v", err)
	}

	// Compare results
	maxDiff, avgDiff := compareFloatTensors(expected, result)

	t.Logf("Max difference: %e", maxDiff)
	t.Logf("Avg difference: %e", avgDiff)

	// Allow small numerical differences due to quantization
	if maxDiff > 1e-4 {
		t.Errorf("Max difference %e exceeds threshold 1e-4", maxDiff)
	}
}

// TestMatMulQ6K_Correctness validates Q6_K fused implementation
func TestMatMulQ6K_Correctness(t *testing.T) {
	M, K, N := 4, 8, 4

	// Create input matrix A (Float32)
	a := NewTensor([]int{M, K}, Float32)
	aData := a.data.([]float32)
	for i := 0; i < M*K; i++ {
		aData[i] = float32(i%10) * 0.1
	}

	// Create weight matrix B (Float32, will be quantized)
	bFloat := NewTensor([]int{K, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := 0; i < K*N; i++ {
		bFloatData[i] = float32(i%20) * 0.5
	}

	// Quantize B to Q6_K
	bQuant, err := QuantizeQ6_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize tensor: %v", err)
	}

	// Method 1: Current approach
	bDequant, err := DequantizeQ6_KTensor(bQuant)
	if err != nil {
		t.Fatalf("Failed to dequantize tensor: %v", err)
	}
	expected := MatMul(a, bDequant)

	// Method 2: Fused approach
	result, err := MatMulQ6K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ6K failed: %v", err)
	}

	// Compare results
	maxDiff, avgDiff := compareFloatTensors(expected, result)

	t.Logf("Max difference: %e", maxDiff)
	t.Logf("Avg difference: %e", avgDiff)

	if maxDiff > 1e-4 {
		t.Errorf("Max difference %e exceeds threshold 1e-4", maxDiff)
	}
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// TestMatMulQ5K_SmallMatrix tests with very small matrices (2x2)
func TestMatMulQ5K_SmallMatrix(t *testing.T) {
	M, K, N := 2, 2, 2

	a := NewTensor([]int{M, K}, Float32)
	aData := a.data.([]float32)
	aData[0], aData[1] = 1.0, 2.0
	aData[2], aData[3] = 3.0, 4.0

	bFloat := NewTensor([]int{K, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	bFloatData[0], bFloatData[1] = 5.0, 6.0
	bFloatData[2], bFloatData[3] = 7.0, 8.0

	bQuant, err := QuantizeQ5_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize: %v", err)
	}

	result, err := MatMulQ5K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ5K failed: %v", err)
	}

	// Verify result shape
	if len(result.shape) != 2 || result.shape[0] != M || result.shape[1] != N {
		t.Errorf("Output shape mismatch: got %v, want [%d,%d]", result.shape, M, N)
	}
}

// TestMatMulQ5K_ZeroMatrix tests with zero-filled matrix
func TestMatMulQ5K_ZeroMatrix(t *testing.T) {
	M, K, N := 4, 8, 4

	a := NewTensor([]int{M, K}, Float32)
	// Leave A as zeros

	bFloat := NewTensor([]int{K, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := range bFloatData {
		bFloatData[i] = 1.0
	}

	bQuant, err := QuantizeQ5_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize: %v", err)
	}

	result, err := MatMulQ5K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ5K failed: %v", err)
	}

	// Result should be all zeros
	resultData := result.data.([]float32)
	for i, val := range resultData {
		if math.Abs(float64(val)) > 1e-6 {
			t.Errorf("Expected zero at index %d, got %f", i, val)
		}
	}
}

// TestMatMulQ5K_Identity tests with identity-like matrix
func TestMatMulQ5K_Identity(t *testing.T) {
	N := 4

	// A = Identity matrix
	a := NewTensor([]int{N, N}, Float32)
	aData := a.data.([]float32)
	for i := 0; i < N; i++ {
		aData[i*N+i] = 1.0
	}

	// B = Some values
	bFloat := NewTensor([]int{N, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := 0; i < N*N; i++ {
		bFloatData[i] = float32(i)
	}

	// Method 1: Reference (dequant + matmul)
	bQuant, err := QuantizeQ5_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize: %v", err)
	}

	bDequant, _ := DequantizeQ5_KTensor(bQuant)
	expected := MatMul(a, bDequant)

	// Method 2: Fused
	result, err := MatMulQ5K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ5K failed: %v", err)
	}

	// Identity × B with fused should match Identity × B with reference
	maxDiff, _ := compareFloatTensors(expected, result)

	if maxDiff > 1e-4 {
		t.Errorf("Identity test failed: max diff %e", maxDiff)
	}
}

// ============================================================================
// Size Variation Tests
// ============================================================================

// TestMatMulQ5K_MediumMatrix tests with medium-sized matrices (64x64)
func TestMatMulQ5K_MediumMatrix(t *testing.T) {
	M, K, N := 64, 64, 64

	a := NewTensor([]int{M, K}, Float32)
	aData := a.data.([]float32)
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}

	bFloat := NewTensor([]int{K, N}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := range bFloatData {
		bFloatData[i] = float32(i%100) * 0.01
	}

	bQuant, err := QuantizeQ5_KTensor(bFloat)
	if err != nil {
		t.Fatalf("Failed to quantize: %v", err)
	}

	// Method 1: Reference
	bDequant, _ := DequantizeQ5_KTensor(bQuant)
	expected := MatMul(a, bDequant)

	// Method 2: Fused
	result, err := MatMulQ5K(a, bQuant)
	if err != nil {
		t.Fatalf("MatMulQ5K failed: %v", err)
	}

	maxDiff, _ := compareFloatTensors(expected, result)
	if maxDiff > 1e-3 { // Slightly higher tolerance for larger matrices
		t.Errorf("Medium matrix test failed: max diff %e", maxDiff)
	}
}

// TestMatMulQ5K_NonSquare tests with non-square matrices
func TestMatMulQ5K_NonSquare(t *testing.T) {
	testCases := []struct {
		name   string
		M, K, N int
	}{
		{"tall", 16, 8, 4},
		{"wide", 4, 8, 16},
		{"rectangular", 32, 64, 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewTensor([]int{tc.M, tc.K}, Float32)
			aData := a.data.([]float32)
			for i := range aData {
				aData[i] = 1.0
			}

			bFloat := NewTensor([]int{tc.K, tc.N}, Float32)
			bFloatData := bFloat.data.([]float32)
			for i := range bFloatData {
				bFloatData[i] = 1.0
			}

			bQuant, err := QuantizeQ5_KTensor(bFloat)
			if err != nil {
				t.Fatalf("Failed to quantize: %v", err)
			}

			result, err := MatMulQ5K(a, bQuant)
			if err != nil {
				t.Fatalf("MatMulQ5K failed: %v", err)
			}

			// Verify shape
			if result.shape[0] != tc.M || result.shape[1] != tc.N {
				t.Errorf("Shape mismatch: got [%d,%d], want [%d,%d]",
					result.shape[0], result.shape[1], tc.M, tc.N)
			}
		})
	}
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// TestMatMulQ5K_NilInputs tests error handling for nil inputs
func TestMatMulQ5K_NilInputs(t *testing.T) {
	a := NewTensor([]int{2, 2}, Float32)

	_, err := MatMulQ5K(nil, a)
	if err == nil {
		t.Error("Expected error for nil first argument")
	}

	_, err = MatMulQ5K(a, nil)
	if err == nil {
		t.Error("Expected error for nil second argument")
	}
}

// TestMatMulQ5K_WrongDType tests error handling for wrong data types
func TestMatMulQ5K_WrongDType(t *testing.T) {
	a := NewTensor([]int{2, 2}, Float16) // Wrong dtype
	b := NewTensor([]int{2, 2}, Float32)
	bQuant, _ := QuantizeQ5_KTensor(b)

	_, err := MatMulQ5K(a, bQuant)
	if err == nil {
		t.Error("Expected error for wrong dtype on first argument")
	}

	aFloat := NewTensor([]int{2, 2}, Float32)
	bFloat := NewTensor([]int{2, 2}, Float32) // Not quantized

	_, err = MatMulQ5K(aFloat, bFloat)
	if err == nil {
		t.Error("Expected error for non-quantized second argument")
	}
}

// TestMatMulQ5K_IncompatibleDimensions tests dimension mismatch
func TestMatMulQ5K_IncompatibleDimensions(t *testing.T) {
	a := NewTensor([]int{4, 8}, Float32)
	bFloat := NewTensor([]int{6, 4}, Float32) // Incompatible: 8 != 6
	bQuant, _ := QuantizeQ5_KTensor(bFloat)

	_, err := MatMulQ5K(a, bQuant)
	if err == nil {
		t.Error("Expected error for incompatible dimensions")
	}
}

// ============================================================================
// Benchmarks
// ============================================================================

// BenchmarkMatMulQ5K_Current benchmarks the current approach (dequant + matmul)
func BenchmarkMatMulQ5K_Current(b *testing.B) {
	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			a := NewTensor([]int{size, size}, Float32)
			aData := a.data.([]float32)
			for i := range aData {
				aData[i] = 1.0
			}

			bFloat := NewTensor([]int{size, size}, Float32)
			bFloatData := bFloat.data.([]float32)
			for i := range bFloatData {
				bFloatData[i] = 1.0
			}

			bQuant, _ := QuantizeQ5_KTensor(bFloat)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Current approach: Dequantize then MatMul
				bDequant, _ := DequantizeQ5_KTensor(bQuant)
				_ = MatMul(a, bDequant)
			}
		})
	}
}

// BenchmarkMatMulQ5K_Fused benchmarks the fused approach
func BenchmarkMatMulQ5K_Fused(b *testing.B) {
	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			a := NewTensor([]int{size, size}, Float32)
			aData := a.data.([]float32)
			for i := range aData {
				aData[i] = 1.0
			}

			bFloat := NewTensor([]int{size, size}, Float32)
			bFloatData := bFloat.data.([]float32)
			for i := range bFloatData {
				bFloatData[i] = 1.0
			}

			bQuant, _ := QuantizeQ5_KTensor(bFloat)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Fused approach
				_, _ = MatMulQ5K(a, bQuant)
			}
		})
	}
}

// BenchmarkMatMulQ6K_Comparison benchmarks Q6_K fused vs current
func BenchmarkMatMulQ6K_Comparison(b *testing.B) {
	size := 128

	a := NewTensor([]int{size, size}, Float32)
	aData := a.data.([]float32)
	for i := range aData {
		aData[i] = 1.0
	}

	bFloat := NewTensor([]int{size, size}, Float32)
	bFloatData := bFloat.data.([]float32)
	for i := range bFloatData {
		bFloatData[i] = 1.0
	}

	bQuant, _ := QuantizeQ6_KTensor(bFloat)

	b.Run("Current", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			bDequant, _ := DequantizeQ6_KTensor(bQuant)
			_ = MatMul(a, bDequant)
		}
	})

	b.Run("Fused", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = MatMulQ6K(a, bQuant)
		}
	})
}

// ============================================================================
// Helper Functions
// ============================================================================

// compareFloatTensors compares two Float32 tensors and returns max and average difference
func compareFloatTensors(a, b *Tensor) (maxDiff, avgDiff float64) {
	if a.DType() != Float32 || b.DType() != Float32 {
		return math.Inf(1), math.Inf(1)
	}

	aData := a.data.([]float32)
	bData := b.data.([]float32)

	if len(aData) != len(bData) {
		return math.Inf(1), math.Inf(1)
	}

	maxDiff = 0.0
	totalDiff := 0.0

	for i := range aData {
		diff := math.Abs(float64(aData[i] - bData[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		totalDiff += diff
	}

	avgDiff = totalDiff / float64(len(aData))
	return maxDiff, avgDiff
}
