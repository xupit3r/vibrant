package tensor

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := Add(a, b)

	expected := []float32{6, 8, 10, 12}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("Add[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestSub(t *testing.T) {
	a := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})
	b := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})

	c := Sub(a, b)

	expected := []float32{4, 4, 4, 4}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("Sub[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{2, 3, 4, 5}, []int{2, 2})

	c := Mul(a, b)

	expected := []float32{2, 6, 12, 20}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("Mul[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestDiv(t *testing.T) {
	a := NewTensorFromData([]float32{10, 20, 30, 40}, []int{2, 2})
	b := NewTensorFromData([]float32{2, 4, 5, 8}, []int{2, 2})

	c := Div(a, b)

	expected := []float32{5, 5, 6, 5}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("Div[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestAddScalar(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})

	c := AddScalar(a, 10)

	expected := []float32{11, 12, 13, 14}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("AddScalar[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMulScalar(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})

	c := MulScalar(a, 3)

	expected := []float32{3, 6, 9, 12}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("MulScalar[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestDivScalar(t *testing.T) {
	a := NewTensorFromData([]float32{10, 20, 30, 40}, []int{2, 2})

	c := DivScalar(a, 10)

	expected := []float32{1, 2, 3, 4}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("DivScalar[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestSum(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4, 5}, []int{5})

	sum := Sum(a)

	expected := float32(15.0)
	if sum != expected {
		t.Errorf("Sum: expected %f, got %f", expected, sum)
	}
}

func TestMean(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4, 5}, []int{5})

	mean := Mean(a)

	expected := float32(3.0)
	if mean != expected {
		t.Errorf("Mean: expected %f, got %f", expected, mean)
	}
}

func TestMax(t *testing.T) {
	a := NewTensorFromData([]float32{1, 5, 3, 2, 4}, []int{5})

	max := Max(a)

	expected := float32(5.0)
	if max != expected {
		t.Errorf("Max: expected %f, got %f", expected, max)
	}
}

func TestMin(t *testing.T) {
	a := NewTensorFromData([]float32{3, 5, 1, 2, 4}, []int{5})

	min := Min(a)

	expected := float32(1.0)
	if min != expected {
		t.Errorf("Min: expected %f, got %f", expected, min)
	}
}

func TestReLU(t *testing.T) {
	a := NewTensorFromData([]float32{-2, -1, 0, 1, 2}, []int{5})

	c := ReLU(a)

	expected := []float32{0, 0, 0, 1, 2}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("ReLU[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestSigmoid(t *testing.T) {
	a := NewTensorFromData([]float32{0.0}, []int{1})

	c := Sigmoid(a)

	// sigmoid(0) = 0.5
	expected := float32(0.5)
	result := c.Data().([]float32)[0]

	if math.Abs(float64(result-expected)) > 1e-6 {
		t.Errorf("Sigmoid(0): expected %f, got %f", expected, result)
	}
}

func TestSiLU(t *testing.T) {
	a := NewTensorFromData([]float32{0.0, 1.0}, []int{2})

	c := SiLU(a)

	data := c.Data().([]float32)

	// SiLU(0) = 0
	if math.Abs(float64(data[0])) > 1e-6 {
		t.Errorf("SiLU(0): expected 0, got %f", data[0])
	}

	// SiLU(1) ≈ 0.731
	expected := float32(0.731)
	if math.Abs(float64(data[1]-expected)) > 0.01 {
		t.Errorf("SiLU(1): expected ~%f, got %f", expected, data[1])
	}
}

func TestGELU(t *testing.T) {
	a := NewTensorFromData([]float32{0.0}, []int{1})

	c := GELU(a)

	// GELU(0) = 0
	result := c.Data().([]float32)[0]
	if math.Abs(float64(result)) > 1e-6 {
		t.Errorf("GELU(0): expected 0, got %f", result)
	}
}

func TestSoftmax(t *testing.T) {
	a := NewTensorFromData([]float32{1.0, 2.0, 3.0}, []int{3})

	c := Softmax(a)

	data := c.Data().([]float32)

	// Check sum is 1
	sum := float32(0)
	for _, val := range data {
		sum += val
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum: expected 1.0, got %f", sum)
	}

	// Check all values are positive
	for i, val := range data {
		if val <= 0 {
			t.Errorf("Softmax[%d]: expected positive value, got %f", i, val)
		}
	}

	// Check values are in ascending order (since input was ascending)
	if !(data[0] < data[1] && data[1] < data[2]) {
		t.Errorf("Softmax: expected ascending values, got %v", data)
	}
}

func TestReshape(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})

	// Reshape from [2, 3] to [3, 2]
	c := Reshape(a, []int{3, 2})

	if len(c.Shape()) != 2 || c.Shape()[0] != 3 || c.Shape()[1] != 2 {
		t.Errorf("Reshape: expected shape [3, 2], got %v", c.Shape())
	}

	// Verify data is shared (not copied)
	if &a.Data().([]float32)[0] != &c.Data().([]float32)[0] {
		t.Error("Reshape should share data, not copy")
	}
}

func TestReshapePanicsOnSizeMismatch(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on size mismatch")
		}
	}()

	Reshape(a, []int{3, 3}) // 4 elements can't be reshaped to 9
}

func TestTranspose(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})

	c := Transpose(a)

	// Check shape
	if len(c.Shape()) != 2 || c.Shape()[0] != 3 || c.Shape()[1] != 2 {
		t.Errorf("Transpose: expected shape [3, 2], got %v", c.Shape())
	}

	// Check values
	// Original: [[1, 2, 3], [4, 5, 6]]
	// Transposed: [[1, 4], [2, 5], [3, 6]]
	if c.At(0, 0) != 1 || c.At(0, 1) != 4 {
		t.Errorf("Transpose[0]: expected [1, 4], got [%f, %f]", c.At(0, 0), c.At(0, 1))
	}
	if c.At(1, 0) != 2 || c.At(1, 1) != 5 {
		t.Errorf("Transpose[1]: expected [2, 5], got [%f, %f]", c.At(1, 0), c.At(1, 1))
	}
	if c.At(2, 0) != 3 || c.At(2, 1) != 6 {
		t.Errorf("Transpose[2]: expected [3, 6], got [%f, %f]", c.At(2, 0), c.At(2, 1))
	}
}

// Benchmarks

func BenchmarkAdd(b *testing.B) {
	a := NewTensor([]int{1024, 1024}, Float32)
	x := NewTensor([]int{1024, 1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Add(a, x)
	}
}

func BenchmarkMul(b *testing.B) {
	a := NewTensor([]int{1024, 1024}, Float32)
	x := NewTensor([]int{1024, 1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Mul(a, x)
	}
}

func BenchmarkReLU(b *testing.B) {
	a := NewTensor([]int{1024, 1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ReLU(a)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	a := NewTensor([]int{1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(a)
	}
}

func BenchmarkTranspose(b *testing.B) {
	a := NewTensor([]int{1024, 1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Transpose(a)
	}
}
