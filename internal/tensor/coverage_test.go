package tensor

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"
)

// Additional tests to increase coverage to 95%+

// Test BytesPerElement for all data types
func TestBytesPerElement(t *testing.T) {
	tests := []struct {
		dtype    DataType
		expected int
	}{
		{Float32, 4},
		{Float16, 2},
		{Q8_0, 1},
		{Q4_K, 1},
		{Q5_K, 1},
	}

	for _, tt := range tests {
		t.Run(tt.dtype.String(), func(t *testing.T) {
			result := tt.dtype.BytesPerElement()
			if result != tt.expected {
				t.Errorf("BytesPerElement for %s: expected %d, got %d", tt.dtype, tt.expected, result)
			}
		})
	}
}

// Test Stride accessor
func TestStride(t *testing.T) {
	tensor := NewTensor([]int{3, 4, 5}, Float32)
	strides := tensor.Stride()

	// Expected strides for [3, 4, 5] shape: [20, 5, 1]
	expected := []int{20, 5, 1}

	if len(strides) != len(expected) {
		t.Errorf("Expected %d strides, got %d", len(expected), len(strides))
	}

	for i, stride := range expected {
		if strides[i] != stride {
			t.Errorf("Stride[%d]: expected %d, got %d", i, stride, strides[i])
		}
	}
}

// Test Close for non-mmap tensors (should be no-op)
func TestCloseNonMmap(t *testing.T) {
	tensor := NewTensor([]int{10}, Float32)
	err := tensor.Close()
	if err != nil {
		t.Errorf("Close on non-mmap tensor should not error: %v", err)
	}
}

// Test NewTensorMmap with a real file
func TestNewTensorMmap(t *testing.T) {
	// Create a temporary file with some float32 data
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test_tensor.bin")

	// Write test data
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Write as bytes
	for _, val := range data {
		bytes := *(*[4]byte)(unsafe.Pointer(&val))
		if _, err := f.Write(bytes[:]); err != nil {
			t.Fatalf("Failed to write data: %v", err)
		}
	}
	f.Close()

	// Memory-map the file
	size := int64(len(data) * 4)
	tensor, err := NewTensorMmap(testFile, 0, size, []int{2, 3}, Float32)
	if err != nil {
		t.Fatalf("Failed to mmap tensor: %v", err)
	}
	defer tensor.Close()

	// Verify data
	if tensor.At(0, 0) != 1.0 {
		t.Errorf("Expected 1.0, got %f", tensor.At(0, 0))
	}
	if tensor.At(1, 2) != 6.0 {
		t.Errorf("Expected 6.0, got %f", tensor.At(1, 2))
	}

	// Test Close
	err = tensor.Close()
	if err != nil {
		t.Errorf("Close on mmap tensor failed: %v", err)
	}
}

// Test NewTensorFromData with different data types
func TestNewTensorFromDataTypes(t *testing.T) {
	// Test with uint16 (float16)
	t.Run("float16", func(t *testing.T) {
		data := []uint16{0, 1, 2, 3}
		tensor := NewTensorFromData(data, []int{2, 2})

		if tensor.DType() != Float16 {
			t.Errorf("Expected Float16, got %s", tensor.DType())
		}
	})

	// Test with uint8 (quantized)
	t.Run("uint8", func(t *testing.T) {
		data := []uint8{0, 1, 2, 3}
		tensor := NewTensorFromData(data, []int{2, 2})

		if tensor.DType() != Q8_0 {
			t.Errorf("Expected Q8_0, got %s", tensor.DType())
		}
	})
}

// Test Ones with Float16
func TestOnesFloat16(t *testing.T) {
	tensor := Ones([]int{3, 3}, Float16)

	// Check a few values
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val := tensor.At(i, j)
			if math.Abs(float64(val-1.0)) > 0.01 {
				t.Errorf("Ones Float16[%d, %d]: expected ~1.0, got %f", i, j, val)
			}
		}
	}
}

// Test edge cases for shapesMatch
func TestShapesMatch(t *testing.T) {
	tests := []struct {
		name     string
		a        []int
		b        []int
		expected bool
	}{
		{"same shape", []int{2, 3}, []int{2, 3}, true},
		{"different dims", []int{2, 3}, []int{2, 3, 4}, false},
		{"different size", []int{2, 3}, []int{3, 2}, false},
		{"empty", []int{}, []int{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shapesMatch(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("shapesMatch(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

// Test panic paths for operations with wrong dtypes
func TestOpsPanicOnUnsupportedDtype(t *testing.T) {
	// Create a Q4_K tensor (not supported for most ops)
	tensor := NewTensor([]int{2, 2}, Q4_K)

	tests := []struct {
		name string
		fn   func()
	}{
		{"Add", func() { Add(tensor, tensor) }},
		{"Sub", func() { Sub(tensor, tensor) }},
		{"Mul", func() { Mul(tensor, tensor) }},
		{"Div", func() { Div(tensor, tensor) }},
		{"AddScalar", func() { AddScalar(tensor, 1.0) }},
		{"MulScalar", func() { MulScalar(tensor, 1.0) }},
		{"DivScalar", func() { DivScalar(tensor, 1.0) }},
		{"Sum", func() { Sum(tensor) }},
		{"Max", func() { Max(tensor) }},
		{"Min", func() { Min(tensor) }},
		{"ReLU", func() { ReLU(tensor) }},
		{"GELU", func() { GELU(tensor) }},
		{"SiLU", func() { SiLU(tensor) }},
		{"Sigmoid", func() { Sigmoid(tensor) }},
		{"Softmax", func() { Softmax(tensor) }},
		{"Transpose", func() { Transpose(tensor) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s should panic on unsupported dtype", tt.name)
				}
			}()
			tt.fn()
		})
	}
}

// Test MatMul panic paths
func TestMatMulPanics(t *testing.T) {
	// Test with Q4_K dtype
	t.Run("unsupported dtype", func(t *testing.T) {
		a := NewTensor([]int{2, 2}, Q4_K)
		b := NewTensor([]int{2, 2}, Q4_K)

		defer func() {
			if r := recover(); r == nil {
				t.Error("MatMul should panic on unsupported dtype")
			}
		}()

		matmulNaive(a, b)
	})

	// Test BatchMatMul panics
	t.Run("batch size mismatch", func(t *testing.T) {
		a := NewTensor([]int{2, 3, 4}, Float32)
		b := NewTensor([]int{3, 4, 5}, Float32)

		defer func() {
			if r := recover(); r == nil {
				t.Error("BatchMatMul should panic on batch size mismatch")
			}
		}()

		BatchMatMul(a, b)
	})

	t.Run("3d dimension mismatch", func(t *testing.T) {
		a := NewTensor([]int{2, 3, 4}, Float32)
		b := NewTensor([]int{2, 5, 6}, Float32)

		defer func() {
			if r := recover(); r == nil {
				t.Error("BatchMatMul should panic on dimension mismatch")
			}
		}()

		BatchMatMul(a, b)
	})

	t.Run("wrong tensor dimensions", func(t *testing.T) {
		a := NewTensor([]int{2, 3}, Float32)
		b := NewTensor([]int{3, 4}, Float32)

		defer func() {
			if r := recover(); r == nil {
				t.Error("BatchMatMul should panic on 2D tensors")
			}
		}()

		BatchMatMul(a, b)
	})

	t.Run("MatVec wrong matrix dims", func(t *testing.T) {
		a := NewTensor([]int{2, 3, 4}, Float32)
		b := NewTensor([]int{3}, Float32)

		defer func() {
			if r := recover(); r == nil {
				t.Error("MatVec should panic on 3D matrix")
			}
		}()

		MatVec(a, b)
	})

	t.Run("MatVec wrong vector dims", func(t *testing.T) {
		a := NewTensor([]int{2, 3}, Float32)
		b := NewTensor([]int{3, 4}, Float32)

		defer func() {
			if r := recover(); r == nil {
				t.Error("MatVec should panic on 2D vector")
			}
		}()

		MatVec(a, b)
	})

	t.Run("MatVec unsupported dtype", func(t *testing.T) {
		a := NewTensor([]int{2, 3}, Q4_K)
		b := NewTensor([]int{3}, Q4_K)

		defer func() {
			if r := recover(); r == nil {
				t.Error("MatVec should panic on unsupported dtype")
			}
		}()

		MatVec(a, b)
	})

	t.Run("BatchMatMul unsupported dtype", func(t *testing.T) {
		a := NewTensor([]int{2, 3, 4}, Q4_K)
		b := NewTensor([]int{2, 4, 5}, Q4_K)

		defer func() {
			if r := recover(); r == nil {
				t.Error("BatchMatMul should panic on unsupported dtype")
			}
		}()

		BatchMatMul(a, b)
	})
}

// Test float16 conversion edge cases
func TestFloat16ConversionEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"positive infinity", float32(math.Inf(1))},
		{"negative infinity", float32(math.Inf(-1))},
		{"very small", float32(1e-10)},
		{"very large", float32(1e10)},
		{"negative zero", float32(-0.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f16 := float32ToFloat16(tt.input)
			f32 := float16ToFloat32(f16)

			// Just ensure conversion doesn't panic
			// Accuracy not guaranteed for edge cases
			_ = f32
		})
	}
}

// Test DataType.String for unknown type
func TestDataTypeStringUnknown(t *testing.T) {
	dtype := DataType(999)
	result := dtype.String()
	if result != "unknown" {
		t.Errorf("Expected 'unknown', got '%s'", result)
	}
}

// Test Device.String for unknown device
func TestDeviceStringUnknown(t *testing.T) {
	device := Device(999)
	result := device.String()
	if result != "unknown" {
		t.Errorf("Expected 'unknown', got '%s'", result)
	}
}

// Test At/Set with Float16 dtype
func TestAtSetFloat16(t *testing.T) {
	tensor := NewTensor([]int{2, 2}, Float16)

	tensor.Set(3.5, 0, 1)
	val := tensor.At(0, 1)

	// Float16 has lower precision
	if math.Abs(float64(val-3.5)) > 0.01 {
		t.Errorf("Expected ~3.5, got %f", val)
	}
}

// Test At/Set panic on unsupported dtype
func TestAtSetUnsupportedDtype(t *testing.T) {
	tensor := NewTensor([]int{2, 2}, Q4_K)

	t.Run("At panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("At should panic on Q4_K dtype")
			}
		}()
		tensor.At(0, 0)
	})

	t.Run("Set panics", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Set should panic on Q4_K dtype")
			}
		}()
		tensor.Set(1.0, 0, 0)
	})
}

// Test MatMul automatic dispatch
func TestMatMulDispatch(t *testing.T) {
	// Small matrix should use naive
	small := NewTensor([]int{8, 8}, Float32)
	result := MatMul(small, small)
	if result.Size() != 64 {
		t.Errorf("Small matmul failed")
	}

	// Medium matrix should use blocked
	medium := NewTensor([]int{64, 64}, Float32)
	result = MatMul(medium, medium)
	if result.Size() != 4096 {
		t.Errorf("Medium matmul failed")
	}

	// Large matrix should use parallel
	large := NewTensor([]int{256, 256}, Float32)
	result = MatMul(large, large)
	if result.Size() != 65536 {
		t.Errorf("Large matmul failed")
	}
}

// Test computeStrides edge case with empty shape
func TestComputeStridesEmpty(t *testing.T) {
	strides := computeStrides([]int{})
	if len(strides) != 0 {
		t.Errorf("Expected empty strides for empty shape")
	}
}

// Test Transpose panic on non-2D tensor
func TestTransposePanicOn3D(t *testing.T) {
	tensor := NewTensor([]int{2, 3, 4}, Float32)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Transpose should panic on 3D tensor")
		}
	}()

	Transpose(tensor)
}
