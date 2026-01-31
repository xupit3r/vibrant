package tensor

import (
	"math"
	"runtime"
	"testing"
)

func TestDetectCPUFeatures(t *testing.T) {
	features := detectCPUFeatures()

	// Verify that features match the architecture
	switch runtime.GOARCH {
	case "amd64":
		if !features.hasAVX2 {
			t.Error("Expected AVX2 on amd64")
		}
		if features.hasNEON {
			t.Error("Expected no NEON on amd64")
		}
	case "arm64":
		if !features.hasNEON {
			t.Error("Expected NEON on arm64")
		}
		if features.hasAVX2 {
			t.Error("Expected no AVX2 on arm64")
		}
	}
}

func TestHasAVX2(t *testing.T) {
	hasAVX2 := HasAVX2()

	if runtime.GOARCH == "amd64" {
		if !hasAVX2 {
			t.Error("Expected AVX2 on amd64")
		}
	} else {
		if hasAVX2 {
			t.Error("Expected no AVX2 on non-amd64")
		}
	}
}

func TestHasNEON(t *testing.T) {
	hasNEON := HasNEON()

	if runtime.GOARCH == "arm64" {
		if !hasNEON {
			t.Error("Expected NEON on arm64")
		}
	} else {
		if hasNEON {
			t.Error("Expected no NEON on non-arm64")
		}
	}
}

func TestGetSIMDInfo(t *testing.T) {
	info := GetSIMDInfo()

	if info == "" {
		t.Error("Expected non-empty SIMD info")
	}

	t.Logf("SIMD Info: %s", info)
}

func TestVectorAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{5, 4, 3, 2, 1}
	dst := make([]float32, 5)

	vectorAdd(dst, a, b)

	expected := []float32{6, 6, 6, 6, 6}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("vectorAdd[%d]: expected %f, got %f", i, expected[i], dst[i])
		}
	}
}

func TestVectorMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{2, 3, 4, 5, 6}
	dst := make([]float32, 5)

	vectorMul(dst, a, b)

	expected := []float32{2, 6, 12, 20, 30}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("vectorMul[%d]: expected %f, got %f", i, expected[i], dst[i])
		}
	}
}

func TestVectorDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}

	result := vectorDotProduct(a, b)

	// 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
	expected := float32(70)

	if result != expected {
		t.Errorf("vectorDotProduct: expected %f, got %f", expected, result)
	}
}

func TestVectorSum(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}

	result := vectorSum(a)

	expected := float32(15)

	if result != expected {
		t.Errorf("vectorSum: expected %f, got %f", expected, result)
	}
}

func TestVectorMax(t *testing.T) {
	a := []float32{1, 5, 3, 2, 4}

	result := vectorMax(a)

	expected := float32(5)

	if result != expected {
		t.Errorf("vectorMax: expected %f, got %f", expected, result)
	}
}

func TestVectorMin(t *testing.T) {
	a := []float32{3, 5, 1, 2, 4}

	result := vectorMin(a)

	expected := float32(1)

	if result != expected {
		t.Errorf("vectorMin: expected %f, got %f", expected, result)
	}
}

func TestVectorReLU(t *testing.T) {
	a := []float32{-2, -1, 0, 1, 2}
	dst := make([]float32, 5)

	vectorReLU(dst, a)

	expected := []float32{0, 0, 0, 1, 2}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("vectorReLU[%d]: expected %f, got %f", i, expected[i], dst[i])
		}
	}
}

func TestVectorScale(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	dst := make([]float32, 5)

	vectorScale(dst, a, 3)

	expected := []float32{3, 6, 9, 12, 15}
	for i := range expected {
		if dst[i] != expected[i] {
			t.Errorf("vectorScale[%d]: expected %f, got %f", i, expected[i], dst[i])
		}
	}
}

func TestMatMulSIMD(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := matmulSIMD(a, b)

	// Expected: [[19, 22], [43, 50]]
	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("matmulSIMD[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulSIMDBlocked(t *testing.T) {
	// Test with larger matrix
	size := 128
	a := NewTensor([]int{size, size}, Float32)
	b := NewTensor([]int{size, size}, Float32)

	// Fill with pattern
	aData := a.Data().([]float32)
	bData := b.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i % 10)
		bData[i] = float32((i + 1) % 10)
	}

	c := matmulSIMDBlocked(a, b)

	// Just verify it completes without panic and has correct shape
	if len(c.Shape()) != 2 || c.Shape()[0] != size || c.Shape()[1] != size {
		t.Errorf("matmulSIMDBlocked: wrong shape %v", c.Shape())
	}
}

func TestMatMulSIMDParallel(t *testing.T) {
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := matmulSIMDParallel(a, b)

	// Expected: [[19, 22], [43, 50]]
	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("matmulSIMDParallel[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestMatMulWithSIMD(t *testing.T) {
	// Test that MatMul dispatches correctly with SIMD
	a := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	b := NewTensorFromData([]float32{5, 6, 7, 8}, []int{2, 2})

	c := MatMul(a, b)

	expected := []float32{19, 22, 43, 50}
	data := c.Data().([]float32)

	for i, val := range expected {
		if data[i] != val {
			t.Errorf("MatMul with SIMD[%d]: expected %f, got %f", i, val, data[i])
		}
	}
}

func TestSIMDOperationsPreserveAccuracy(t *testing.T) {
	// Verify SIMD operations produce same results as naive implementations
	size := 1000
	a := NewTensor([]int{size}, Float32)
	b := NewTensor([]int{size}, Float32)

	aData := a.Data().([]float32)
	bData := b.Data().([]float32)

	for i := 0; i < size; i++ {
		aData[i] = float32(i) * 0.1
		bData[i] = float32(i) * 0.2
	}

	// Test dot product
	dotSIMD := vectorDotProduct(aData, bData)

	// Compute naive dot product
	dotNaive := float32(0)
	for i := 0; i < size; i++ {
		dotNaive += aData[i] * bData[i]
	}

	// Should be very close (allowing for floating point error)
	// With larger sums and different accumulation order, we need a reasonable tolerance
	// The difference is due to floating point accumulation order (4-way vs linear)
	tolerance := 1.0 // Allow up to 1.0 difference for large sums
	if math.Abs(float64(dotSIMD-dotNaive)) > tolerance {
		t.Errorf("Dot product accuracy: SIMD=%f, Naive=%f, diff=%f, tolerance=%f",
			dotSIMD, dotNaive, dotSIMD-dotNaive, tolerance)
	}
}

// Benchmarks to verify SIMD speedup

func BenchmarkVectorAdd_SIMD(b *testing.B) {
	size := 1024 * 1024
	a := make([]float32, size)
	x := make([]float32, size)
	dst := make([]float32, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectorAdd(dst, a, x)
	}
}

func BenchmarkVectorDotProduct_SIMD(b *testing.B) {
	size := 1024
	a := make([]float32, size)
	x := make([]float32, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectorDotProduct(a, x)
	}
}

func BenchmarkMatMulSIMD_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulSIMD(a, x)
	}
}

func BenchmarkMatMulSIMDParallel_256x256(b *testing.B) {
	a := NewTensor([]int{256, 256}, Float32)
	x := NewTensor([]int{256, 256}, Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matmulSIMDParallel(a, x)
	}
}

func BenchmarkMatMulComparison_SIMD_vs_Naive(b *testing.B) {
	size := 256
	a := NewTensor([]int{size, size}, Float32)
	x := NewTensor([]int{size, size}, Float32)

	b.Run("naive", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matmulNaive(a, x)
		}
	})

	b.Run("simd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matmulSIMD(a, x)
		}
	})

	b.Run("simd_parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matmulSIMDParallel(a, x)
		}
	})
}
