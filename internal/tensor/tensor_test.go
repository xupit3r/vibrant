package tensor

import (
	"math"
	"testing"
)

func TestNewTensor(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		dtype DataType
	}{
		{"1D float32", []int{10}, Float32},
		{"2D float32", []int{3, 4}, Float32},
		{"3D float32", []int{2, 3, 4}, Float32},
		{"2D float16", []int{5, 5}, Float16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape, tt.dtype)

			// Check shape
			if len(tensor.Shape()) != len(tt.shape) {
				t.Errorf("expected %d dimensions, got %d", len(tt.shape), len(tensor.Shape()))
			}

			for i, dim := range tt.shape {
				if tensor.Shape()[i] != dim {
					t.Errorf("dimension %d: expected %d, got %d", i, dim, tensor.Shape()[i])
				}
			}

			// Check dtype
			if tensor.DType() != tt.dtype {
				t.Errorf("expected dtype %s, got %s", tt.dtype, tensor.DType())
			}

			// Check device
			if tensor.Device() != CPU {
				t.Errorf("expected CPU device, got %s", tensor.Device())
			}

			// Check size
			expectedSize := 1
			for _, dim := range tt.shape {
				expectedSize *= dim
			}
			if tensor.Size() != expectedSize {
				t.Errorf("expected size %d, got %d", expectedSize, tensor.Size())
			}
		})
	}
}

func TestComputeStrides(t *testing.T) {
	tests := []struct {
		name           string
		shape          []int
		expectedStride []int
	}{
		{"1D", []int{10}, []int{1}},
		{"2D", []int{3, 4}, []int{4, 1}},
		{"3D", []int{2, 3, 4}, []int{12, 4, 1}},
		{"4D", []int{2, 3, 4, 5}, []int{60, 20, 5, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strides := computeStrides(tt.shape)

			if len(strides) != len(tt.expectedStride) {
				t.Fatalf("expected %d strides, got %d", len(tt.expectedStride), len(strides))
			}

			for i, expected := range tt.expectedStride {
				if strides[i] != expected {
					t.Errorf("stride[%d]: expected %d, got %d", i, expected, strides[i])
				}
			}
		})
	}
}

func TestZeros(t *testing.T) {
	tensor := Zeros([]int{3, 4}, Float32)

	// Check all elements are zero
	data := tensor.Data().([]float32)
	for i, val := range data {
		if val != 0.0 {
			t.Errorf("element %d: expected 0.0, got %f", i, val)
		}
	}
}

func TestOnes(t *testing.T) {
	tensor := Ones([]int{3, 4}, Float32)

	// Check all elements are one
	data := tensor.Data().([]float32)
	for i, val := range data {
		if val != 1.0 {
			t.Errorf("element %d: expected 1.0, got %f", i, val)
		}
	}
}

func TestAtSet(t *testing.T) {
	tensor := NewTensor([]int{3, 4}, Float32)

	// Test Set
	tensor.Set(42.0, 1, 2)

	// Test At
	val := tensor.At(1, 2)
	if val != 42.0 {
		t.Errorf("expected 42.0, got %f", val)
	}

	// Test all other values are zero
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if i == 1 && j == 2 {
				continue
			}
			if tensor.At(i, j) != 0.0 {
				t.Errorf("element [%d,%d]: expected 0.0, got %f", i, j, tensor.At(i, j))
			}
		}
	}
}

func TestAtSetMultipleDimensions(t *testing.T) {
	tensor := NewTensor([]int{2, 3, 4}, Float32)

	// Set values
	tensor.Set(1.0, 0, 0, 0)
	tensor.Set(2.0, 0, 1, 2)
	tensor.Set(3.0, 1, 2, 3)

	// Get values
	if tensor.At(0, 0, 0) != 1.0 {
		t.Errorf("expected 1.0, got %f", tensor.At(0, 0, 0))
	}
	if tensor.At(0, 1, 2) != 2.0 {
		t.Errorf("expected 2.0, got %f", tensor.At(0, 1, 2))
	}
	if tensor.At(1, 2, 3) != 3.0 {
		t.Errorf("expected 3.0, got %f", tensor.At(1, 2, 3))
	}
}

func TestNewTensorFromData(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	shape := []int{2, 3}

	tensor := NewTensorFromData(data, shape)

	// Check shape
	if len(tensor.Shape()) != len(shape) {
		t.Errorf("expected %d dimensions, got %d", len(shape), len(tensor.Shape()))
	}

	// Check values
	if tensor.At(0, 0) != 1.0 {
		t.Errorf("expected 1.0, got %f", tensor.At(0, 0))
	}
	if tensor.At(0, 1) != 2.0 {
		t.Errorf("expected 2.0, got %f", tensor.At(0, 1))
	}
	if tensor.At(1, 2) != 6.0 {
		t.Errorf("expected 6.0, got %f", tensor.At(1, 2))
	}
}

func TestFloat16Conversion(t *testing.T) {
	tests := []float32{
		0.0, 1.0, -1.0, 2.5, -2.5, 0.5, -0.5, 100.0, -100.0,
	}

	for _, val := range tests {
		t.Run("", func(t *testing.T) {
			// Convert to float16 and back
			f16 := float32ToFloat16(val)
			result := float16ToFloat32(f16)

			// Check within tolerance (float16 has lower precision)
			tolerance := float32(0.01)
			if math.Abs(float64(result-val)) > float64(tolerance) {
				t.Errorf("conversion error for %f: got %f (diff=%f)", val, result, result-val)
			}
		})
	}
}

func TestDataTypeString(t *testing.T) {
	tests := []struct {
		dtype    DataType
		expected string
	}{
		{Float32, "float32"},
		{Float16, "float16"},
		{Q4_K, "q4_k"},
		{Q5_K, "q5_k"},
		{Q8_0, "q8_0"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if tt.dtype.String() != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, tt.dtype.String())
			}
		})
	}
}

func TestDeviceString(t *testing.T) {
	tests := []struct {
		device   Device
		expected string
	}{
		{CPU, "cpu"},
		{GPU, "gpu"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if tt.device.String() != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, tt.device.String())
			}
		})
	}
}

func TestTensorNumDims(t *testing.T) {
	tests := []struct {
		shape        []int
		expectedDims int
	}{
		{[]int{10}, 1},
		{[]int{3, 4}, 2},
		{[]int{2, 3, 4}, 3},
		{[]int{2, 3, 4, 5}, 4},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			tensor := NewTensor(tt.shape, Float32)
			if tensor.NumDims() != tt.expectedDims {
				t.Errorf("expected %d dims, got %d", tt.expectedDims, tensor.NumDims())
			}
		})
	}
}

func TestAtSetPanicsOnInvalidIndices(t *testing.T) {
	tensor := NewTensor([]int{3, 4}, Float32)

	// Test panic on wrong number of indices
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on wrong number of indices")
		}
	}()

	tensor.At(1) // Should panic (need 2 indices)
}

func TestSetPanicsOnOutOfBounds(t *testing.T) {
	tensor := NewTensor([]int{3, 4}, Float32)

	// Test panic on out of bounds
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on out of bounds index")
		}
	}()

	tensor.Set(42.0, 5, 2) // Should panic (first dim is 0-2)
}

// Benchmark tests
func BenchmarkNewTensor(b *testing.B) {
	shape := []int{1024, 1024}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewTensor(shape, Float32)
	}
}

func BenchmarkTensorAt(b *testing.B) {
	tensor := NewTensor([]int{100, 100}, Float32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.At(50, 50)
	}
}

func BenchmarkTensorSet(b *testing.B) {
	tensor := NewTensor([]int{100, 100}, Float32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.Set(42.0, 50, 50)
	}
}
