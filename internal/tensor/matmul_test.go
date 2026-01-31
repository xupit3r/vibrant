package tensor

import (
	"math"
	"testing"
)

func TestMatMulSmall(t *testing.T) {
	// Simple 2x2 matrix multiplication
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := MatMul(a, b)

	// Expected result:
	// [1*5 + 2*7,  1*6 + 2*8]   =  [19, 22]
	// [3*5 + 4*7,  3*6 + 4*8]      [43, 50]
	expected := []float32{19, 22, 43, 50}

	if len(c.shape) != 2 || c.shape[0] != 2 || c.shape[1] != 2 {
		t.Errorf("Expected shape [2, 2], got %v", c.shape)
	}

	data := c.Data().([]float32)
	for i, val := range expected {
		if data[i] != val {
			t.Errorf("MatMul[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulRectangular(t *testing.T) {
	// 3x2 @ 2x4 = 3x4
	a := NewTensorFromData([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, []int{3, 2})

	b := NewTensorFromData([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int{2, 4})

	c := MatMul(a, b)

	if len(c.shape) != 2 || c.shape[0] != 3 || c.shape[1] != 4 {
		t.Errorf("Expected shape [3, 4], got %v", c.shape)
	}

	// Verify a few key values
	// Row 0: [1*1 + 2*5, 1*2 + 2*6, 1*3 + 2*7, 1*4 + 2*8] = [11, 14, 17, 20]
	if c.At(0, 0) != 11 || c.At(0, 1) != 14 || c.At(0, 2) != 17 || c.At(0, 3) != 20 {
		t.Errorf("Row 0 incorrect: got [%f, %f, %f, %f]",
			c.At(0, 0), c.At(0, 1), c.At(0, 2), c.At(0, 3))
	}
}

func TestMatMulNaive(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := matmulNaive(a, b)

	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("matmulNaive[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulBlocked(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := matmulBlocked(a, b)

	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("matmulBlocked[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulParallel(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := matmulParallel(a, b)

	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("matmulParallel[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulLarge(t *testing.T) {
	// Test larger matrix (64x64) to ensure all implementations work
	size := 64
	a := NewTensor([]int{size, size}, Float32)
	b := NewTensor([]int{size, size}, Float32)

	// Fill with simple pattern
	aData := a.Data().([]float32)
	bData := b.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i % 10)
		bData[i] = float32((i + 1) % 10)
	}

	// All implementations should give same result
	cNaive := matmulNaive(a, b)
	cBlocked := matmulBlocked(a, b)
	cParallel := matmulParallel(a, b)

	naiveData := cNaive.Data().([]float32)
	blockedData := cBlocked.Data().([]float32)
	parallelData := cParallel.Data().([]float32)

	tolerance := float32(1e-4)

	// Compare naive vs blocked
	for i := range naiveData {
		if math.Abs(float64(naiveData[i]-blockedData[i])) > float64(tolerance) {
			t.Errorf("naive vs blocked mismatch at %d: %f vs %f", i, naiveData[i], blockedData[i])
			break
		}
	}

	// Compare naive vs parallel
	for i := range naiveData {
		if math.Abs(float64(naiveData[i]-parallelData[i])) > float64(tolerance) {
			t.Errorf("naive vs parallel mismatch at %d: %f vs %f", i, naiveData[i], parallelData[i])
			break
		}
	}
}

func TestMatVec(t *testing.T) {
	// 3x4 @ 4 = 3
	a := NewTensorFromData([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, []int{3, 4})

	b := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4})

	c := MatVec(a, b)

	// Expected:
	// Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
	// Row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70
	// Row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110
	expected := []float32{30, 70, 110}

	if len(c.shape) != 1 || c.shape[0] != 3 {
		t.Errorf("Expected shape [3], got %v", c.shape)
	}

	data := c.Data().([]float32)
	for i, val := range expected {
		if data[i] != val {
			t.Errorf("MatVec[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestBatchMatMul(t *testing.T) {
	// Batch size 2, 2x2 matrices
	// Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
	// Batch 1: [[9, 10], [11, 12]] @ [[13, 14], [15, 16]]
	a := NewTensorFromData([]float32{
		1, 2, 3, 4, // Batch 0
		9, 10, 11, 12, // Batch 1
	}, []int{2, 2, 2})

	b := NewTensorFromData([]float32{
		5, 6, 7, 8, // Batch 0
		13, 14, 15, 16, // Batch 1
	}, []int{2, 2, 2})

	c := BatchMatMul(a, b)

	if len(c.shape) != 3 || c.shape[0] != 2 || c.shape[1] != 2 || c.shape[2] != 2 {
		t.Errorf("Expected shape [2, 2, 2], got %v", c.shape)
	}

	data := c.Data().([]float32)

	// Batch 0 expected: [[19, 22], [43, 50]]
	if data[0] != 19 || data[1] != 22 || data[2] != 43 || data[3] != 50 {
		t.Errorf("Batch 0 incorrect: got [%f, %f, %f, %f]", data[0], data[1], data[2], data[3])
	}

	// Batch 1 expected: [[9*13 + 10*15, 9*14 + 10*16], [11*13 + 12*15, 11*14 + 12*16]]
	//                   [[267, 286], [323, 346]]
	if data[4] != 267 || data[5] != 286 || data[6] != 323 || data[7] != 346 {
		t.Errorf("Batch 1 incorrect: got [%f, %f, %f, %f]", data[4], data[5], data[6], data[7])
	}
}

func TestMatMulPanicsOn1D(t *testing.T) {
	a := NewTensor([]int{10}, Float32)
	b := NewTensor([]int{10}, Float32)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic on 1D tensors")
		}
	}()

	MatMul(a, b)
}

func TestMatMulPanicsOnDimensionMismatch(t *testing.T) {
	a := NewTensor([]int{3, 4}, Float32)
	b := NewTensor([]int{5, 6}, Float32) // K doesn't match

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic on dimension mismatch")
		}
	}()

	MatMul(a, b)
}

func TestMatVecPanicsOnShapeMismatch(t *testing.T) {
	a := NewTensor([]int{3, 4}, Float32)
	b := NewTensor([]int{5}, Float32) // Wrong size

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic on dimension mismatch")
		}
	}()

	MatVec(a, b)
}

// Benchmarks

func BenchmarkMatMul_Small_64x64(b *testing.B) {
	a := NewTensor([]int{64, 64}, Float32)
	x := NewTensor([]int{64, 64}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, x)
	}
}

func BenchmarkMatMul_Medium_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, x)
	}
}

func BenchmarkMatMul_Large_512x512(b *testing.B) {
	a := NewTensor([]int{512, 512}, Float32)
	x := NewTensor([]int{512, 512}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(a, x)
	}
}

func BenchmarkMatMulNaive_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulNaive(a, x)
	}
}

func BenchmarkMatMulBlocked_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulBlocked(a, x)
	}
}

func BenchmarkMatMulParallel_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulParallel(a, x)
	}
}

func BenchmarkMatVec_1024x1024(b *testing.B) {
	a := NewTensor([]int{1024, 1024}, Float32)
	x := NewTensor([]int{1024}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(a, x)
	}
}

func BenchmarkBatchMatMul_Batch8_128x128(b *testing.B) {
	a := NewTensor([]int{8, 128, 128}, Float32)
	x := NewTensor([]int{8, 128, 128}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchMatMul(a, x)
	}
}

// Comparative benchmark: naive vs blocked vs parallel
func BenchmarkMatMulComparison(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		a := NewTensor([]int{size, size}, Float32)
		x := NewTensor([]int{size, size}, Float32)

		b.Run("naive_"+string(rune(size)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				matmulNaive(a, x)
			}
		})

		b.Run("blocked_"+string(rune(size)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				matmulBlocked(a, x)
			}
		})

		b.Run("parallel_"+string(rune(size)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				matmulParallel(a, x)
			}
		})
	}
}
