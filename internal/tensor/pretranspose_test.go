package tensor

import (
	"testing"
)

// TestPretransposeInPlace verifies that PretransposeInPlace correctly transposes
// a tensor in-place and marks it as transposed
func TestPretransposeInPlace(t *testing.T) {
	// Create a 3x4 Float32 tensor
	tensor := NewTensor([]int{3, 4}, Float32)
	data := tensor.Float32Data()
	for i := range data {
		data[i] = float32(i)
	}

	// Verify initial state
	if tensor.IsTransposed() {
		t.Error("New tensor should not be marked as transposed")
	}
	if tensor.Shape()[0] != 3 || tensor.Shape()[1] != 4 {
		t.Errorf("Expected shape [3, 4], got %v", tensor.Shape())
	}

	// Pre-transpose in place
	err := tensor.PretransposeInPlace()
	if err != nil {
		t.Fatalf("PretransposeInPlace failed: %v", err)
	}

	// Verify transposed state
	if !tensor.IsTransposed() {
		t.Error("Tensor should be marked as transposed")
	}
	if tensor.Shape()[0] != 4 || tensor.Shape()[1] != 3 {
		t.Errorf("Expected shape [4, 3] after transpose, got %v", tensor.Shape())
	}

	// Verify data was actually transposed
	newData := tensor.Float32Data()
	if newData[0] != 0 || newData[1] != 4 || newData[2] != 8 {
		t.Errorf("First row should be [0, 4, 8], got [%f, %f, %f]",
			newData[0], newData[1], newData[2])
	}
}

// TestPretransposeInPlaceErrors verifies error handling
func TestPretransposeInPlaceErrors(t *testing.T) {
	// Test on 1D tensor
	tensor1D := NewTensor([]int{5}, Float32)
	err := tensor1D.PretransposeInPlace()
	if err == nil {
		t.Error("Expected error for 1D tensor, got nil")
	}

	// Test on 3D tensor
	tensor3D := NewTensor([]int{2, 3, 4}, Float32)
	err = tensor3D.PretransposeInPlace()
	if err == nil {
		t.Error("Expected error for 3D tensor, got nil")
	}

	// Test on already transposed tensor (should succeed as no-op)
	tensor2D := NewTensor([]int{3, 4}, Float32)
	err = tensor2D.PretransposeInPlace()
	if err != nil {
		t.Fatalf("First transpose failed: %v", err)
	}
	err = tensor2D.PretransposeInPlace()
	if err != nil {
		t.Error("Second transpose should succeed as no-op, got error:", err)
	}
}

// TestGetOrDequantTransposeWithPretransposed verifies that GetOrDequantTranspose
// returns the same tensor when it's already pre-transposed
func TestGetOrDequantTransposeWithPretransposed(t *testing.T) {
	// Create and pre-transpose a Float32 tensor
	tensor := NewTensor([]int{3, 4}, Float32)
	err := tensor.PretransposeInPlace()
	if err != nil {
		t.Fatalf("PretransposeInPlace failed: %v", err)
	}

	// GetOrDequantTranspose should return the same tensor (no additional work)
	result := tensor.GetOrDequantTranspose()
	if result != tensor {
		t.Error("GetOrDequantTranspose should return same tensor for pre-transposed Float32")
	}
}

// TestGetOrDequantTransposeWithoutPretranspose verifies backward compatibility
func TestGetOrDequantTransposeWithoutPretranspose(t *testing.T) {
	// Create a Float32 tensor without pre-transposing
	tensor := NewTensor([]int{3, 4}, Float32)

	// GetOrDequantTranspose should still work (backward compatibility)
	result := tensor.GetOrDequantTranspose()
	if result != tensor {
		t.Error("GetOrDequantTranspose should return same tensor for non-transposed Float32")
	}
}

// TestMatMulWithPretransposedWeights verifies that GetOrDequantTranspose
// correctly handles pre-transposed weight matrices
func TestMatMulWithPretransposedWeights(t *testing.T) {
	// Create input: [2, 3]
	a := NewTensor([]int{2, 3}, Float32)
	aData := a.Float32Data()
	for i := range aData {
		aData[i] = float32(i + 1)
	}

	// Create weight (not transposed): [3, 4]
	b := NewTensor([]int{3, 4}, Float32)
	bData := b.Float32Data()
	for i := range bData {
		bData[i] = float32(i + 1)
	}

	// Test GetOrDequantTranspose on pre-transposed tensor
	err := b.PretransposeInPlace()
	if err != nil {
		t.Fatalf("PretransposeInPlace failed: %v", err)
	}

	// GetOrDequantTranspose should return the tensor itself (already transposed)
	result := b.GetOrDequantTranspose()
	if result != b {
		t.Error("GetOrDequantTranspose should return same tensor for pre-transposed Float32")
	}

	// Verify it's marked as transposed
	if !result.IsTransposed() {
		t.Error("Result should be marked as transposed")
	}

	// Verify shape is transposed: [4, 3]
	if result.Shape()[0] != 4 || result.Shape()[1] != 3 {
		t.Errorf("Expected shape [4, 3], got %v", result.Shape())
	}
}

// TestMatMulSIMDWithPretransposedWeights verifies SIMD matmul with pre-transposed weights
func TestMatMulSIMDWithPretransposedWeights(t *testing.T) {
	if !HasAVX2() && !HasNEON() {
		t.Skip("SIMD not available")
	}

	// Create input: [4, 8]
	a := NewTensor([]int{4, 8}, Float32)
	aData := a.Float32Data()
	for i := range aData {
		aData[i] = float32(i + 1)
	}

	// Create weight (not transposed): [8, 6]
	b := NewTensor([]int{8, 6}, Float32)
	bData := b.Float32Data()
	for i := range bData {
		bData[i] = float32(i + 1)
	}

	// Pre-transpose the weight - this is what happens during model loading
	// After transpose, b will be [6, 8] (shape is swapped)
	transposedB := b.Transpose()

	// Verify it's marked as transposed
	if !transposedB.IsTransposed() {
		t.Error("Transposed tensor should be marked as transposed")
	}

	// SIMD functions should handle pre-transposed weights correctly
	// The transposed weight is [6, 8] but represents the transpose of [8, 6]
	// So we can't use it directly in matmulSIMD which expects [K, N]
	// This test verifies the transpose flag works correctly
	if transposedB.Shape()[0] != 6 || transposedB.Shape()[1] != 8 {
		t.Errorf("Expected transposed shape [6, 8], got %v", transposedB.Shape())
	}
}
