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

// ============================================================================
// Phase 2: Optimized Implementation Tests
// ============================================================================

// TestMatMulQ5KOptimized_Correctness validates optimized version matches reference
func TestMatMulQ5KOptimized_Correctness(t *testing.T) {
size := 64

// Create input tensors
a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%100) / 10.0
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%50) / 5.0
}

// Quantize B
bQuant, err := QuantizeQ5_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

// Reference: Naive fused implementation
expected, err := MatMulQ5K(a, bQuant)
if err != nil {
t.Fatalf("Reference implementation failed: %v", err)
}

// Test: Optimized fused implementation
result, err := MatMulQ5KOptimized(a, bQuant)
if err != nil {
t.Fatalf("Optimized implementation failed: %v", err)
}

// Compare results
maxDiff, avgDiff := compareFloatTensors(expected, result)
t.Logf("Optimized vs Reference: Max diff = %.2e, Avg diff = %.2e", maxDiff, avgDiff)

if maxDiff > 1e-4 {
t.Errorf("Max difference %.2e exceeds threshold 1e-4", maxDiff)
}
}

// TestMatMulQ6KOptimized_Correctness validates optimized Q6_K version
func TestMatMulQ6KOptimized_Correctness(t *testing.T) {
size := 64

a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%100) / 10.0
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%50) / 5.0
}

bQuant, err := QuantizeQ6_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

expected, err := MatMulQ6K(a, bQuant)
if err != nil {
t.Fatalf("Reference failed: %v", err)
}

result, err := MatMulQ6KOptimized(a, bQuant)
if err != nil {
t.Fatalf("Optimized failed: %v", err)
}

maxDiff, avgDiff := compareFloatTensors(expected, result)
t.Logf("Optimized vs Reference: Max diff = %.2e, Avg diff = %.2e", maxDiff, avgDiff)

if maxDiff > 1e-4 {
t.Errorf("Max difference %.2e exceeds threshold 1e-4", maxDiff)
}
}

// TestMatMulQ4KOptimized_Correctness validates optimized Q4_K version
func TestMatMulQ4KOptimized_Correctness(t *testing.T) {
size := 64

a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%100) / 10.0
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%50) / 5.0
}

bQuant, err := QuantizeQ4_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

// Reference: naive approach (dequant + matmul)
bDequant, err := DequantizeQ4_KTensor(bQuant)
if err != nil {
t.Fatalf("Failed to dequantize: %v", err)
}
expected := MatMul(a, bDequant)

// Test: optimized fused
result, err := MatMulQ4KOptimized(a, bQuant)
if err != nil {
t.Fatalf("Optimized failed: %v", err)
}

maxDiff, avgDiff := compareFloatTensors(expected, result)
t.Logf("Optimized vs Reference: Max diff = %.2e, Avg diff = %.2e", maxDiff, avgDiff)

if maxDiff > 1e-4 {
t.Errorf("Max difference %.2e exceeds threshold 1e-4", maxDiff)
}
}

// ============================================================================
// Phase 2: Performance Benchmarks
// ============================================================================

// BenchmarkMatMulQ5K_Optimized benchmarks the optimized Q5_K implementation
func BenchmarkMatMulQ5K_Optimized(b *testing.B) {
sizes := []int{64, 128, 256}

for _, size := range sizes {
b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
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
_, _ = MatMulQ5KOptimized(a, bQuant)
}
})
}
}

// BenchmarkMatMulQ5K_Comparison compares all three approaches
func BenchmarkMatMulQ5K_Comparison(b *testing.B) {
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

bQuant, _ := QuantizeQ5_KTensor(bFloat)

b.Run("Current_DequantThenMatMul", func(b *testing.B) {
b.ReportAllocs()
for i := 0; i < b.N; i++ {
bDequant, _ := DequantizeQ5_KTensor(bQuant)
_ = MatMul(a, bDequant)
}
})

b.Run("Naive_Fused", func(b *testing.B) {
b.ReportAllocs()
for i := 0; i < b.N; i++ {
_, _ = MatMulQ5K(a, bQuant)
}
})

b.Run("Optimized_Fused", func(b *testing.B) {
b.ReportAllocs()
for i := 0; i < b.N; i++ {
_, _ = MatMulQ5KOptimized(a, bQuant)
}
})
}

// Test Phase 3: Block-cached implementations

func TestMatMulQ5KBlocked_Correctness(t *testing.T) {
sizes := []int{2, 64, 128}

for _, size := range sizes {
t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
// Create test matrices
a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%10) + 0.5
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%7) + 0.3
}

// Quantize B
bQuant, err := QuantizeQ5_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

// Compute reference result using naive implementation
refResult, err := MatMulQ5K(a, bQuant)
if err != nil {
t.Fatalf("Failed naive matmul: %v", err)
}

// Compute blocked result
blockedResult, err := MatMulQ5KBlocked(a, bQuant)
if err != nil {
t.Fatalf("Failed blocked matmul: %v", err)
}

// Compare results
refData := refResult.data.([]float32)
blockedData := blockedResult.data.([]float32)

maxDiff := float32(0.0)
for i := range refData {
diff := float32(math.Abs(float64(refData[i] - blockedData[i])))
if diff > maxDiff {
maxDiff = diff
}
}

// Should be bit-exact since we use same dequantization
if maxDiff > 1e-5 {
t.Errorf("Size %d: Max difference %.2e exceeds tolerance", size, maxDiff)
}
})
}
}

func TestMatMulQ6KBlocked_Correctness(t *testing.T) {
size := 64

a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%10) + 0.5
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%7) + 0.3
}

bQuant, err := QuantizeQ6_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

refResult, err := MatMulQ6K(a, bQuant)
if err != nil {
t.Fatalf("Failed naive matmul: %v", err)
}

blockedResult, err := MatMulQ6KBlocked(a, bQuant)
if err != nil {
t.Fatalf("Failed blocked matmul: %v", err)
}

refData := refResult.data.([]float32)
blockedData := blockedResult.data.([]float32)

maxDiff := float32(0.0)
for i := range refData {
diff := float32(math.Abs(float64(refData[i] - blockedData[i])))
if diff > maxDiff {
maxDiff = diff
}
}

if maxDiff > 1e-5 {
t.Errorf("Max difference %.2e exceeds tolerance", maxDiff)
}
}

func TestMatMulQ4KBlocked_Correctness(t *testing.T) {
size := 64

a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%10) + 0.5
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%7) + 0.3
}

bQuant, err := QuantizeQ4_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

refResult, err := MatMulQ4KOptimized(a, bQuant)
if err != nil {
t.Fatalf("Failed naive matmul: %v", err)
}

blockedResult, err := MatMulQ4KBlocked(a, bQuant)
if err != nil {
t.Fatalf("Failed blocked matmul: %v", err)
}

refData := refResult.data.([]float32)
blockedData := blockedResult.data.([]float32)

maxDiff := float32(0.0)
for i := range refData {
diff := float32(math.Abs(float64(refData[i] - blockedData[i])))
if diff > maxDiff {
maxDiff = diff
}
}

if maxDiff > 1e-5 {
t.Errorf("Max difference %.2e exceeds tolerance", maxDiff)
}
}

// BenchmarkMatMulQ5K_Blocked benchmarks block-cached implementation
func BenchmarkMatMulQ5K_Blocked(b *testing.B) {
sizes := []int{64, 128, 256}

for _, size := range sizes {
b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
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
_, _ = MatMulQ5KBlocked(a, bQuant)
}
})
}
}

// Test Phase 3 V2: Improved block-cached implementations

func TestMatMulQ5KBlockedV2_Correctness(t *testing.T) {
sizes := []int{2, 64, 128}

for _, size := range sizes {
t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
a := NewTensor([]int{size, size}, Float32)
aData := a.data.([]float32)
for i := range aData {
aData[i] = float32(i%10) + 0.5
}

bFloat := NewTensor([]int{size, size}, Float32)
bFloatData := bFloat.data.([]float32)
for i := range bFloatData {
bFloatData[i] = float32(i%7) + 0.3
}

bQuant, err := QuantizeQ5_KTensor(bFloat)
if err != nil {
t.Fatalf("Failed to quantize: %v", err)
}

refResult, err := MatMulQ5K(a, bQuant)
if err != nil {
t.Fatalf("Failed naive matmul: %v", err)
}

v2Result, err := MatMulQ5KBlockedV2(a, bQuant)
if err != nil {
t.Fatalf("Failed V2 matmul: %v", err)
}

refData := refResult.data.([]float32)
v2Data := v2Result.data.([]float32)

maxDiff := float32(0.0)
for i := range refData {
diff := float32(math.Abs(float64(refData[i] - v2Data[i])))
if diff > maxDiff {
maxDiff = diff
}
}

if maxDiff > 1e-5 {
t.Errorf("Size %d: Max difference %.2e exceeds tolerance", size, maxDiff)
}
})
}
}

func BenchmarkMatMulQ5K_BlockedV2(b *testing.B) {
sizes := []int{64, 128, 256}

for _, size := range sizes {
b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
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
_, _ = MatMulQ5KBlockedV2(a, bQuant)
}
})
}
}
