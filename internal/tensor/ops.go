package tensor

import (
	"fmt"
	"math"
)

// Element-wise Operations

// Add performs element-wise addition: C = A + B
// Supports GPU acceleration when both tensors are on GPU
func Add(a, b *Tensor) *Tensor {
	if !shapesMatch(a.shape, b.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.shape, b.shape))
	}

	// Try GPU path first
	if a.IsOnGPU() && b.IsOnGPU() {
		result := addGPU(a, b)
		if result != nil {
			return result
		}
	}

	// CPU fallback — ensure GPU tensors have CPU data available
	if a.IsOnGPU() {
		if _, err := a.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Add: failed to get CPU data for a: %v", err))
		}
	}
	if b.IsOnGPU() {
		if _, err := b.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Add: failed to get CPU data for b: %v", err))
		}
	}

	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)
		// Use SIMD-optimized vector addition
		vectorAdd(cData, aData, bData)
	default:
		panic(fmt.Sprintf("Add not implemented for dtype %s", a.dtype))
	}

	return result
}

// Sub performs element-wise subtraction: C = A - B
func Sub(a, b *Tensor) *Tensor {
	if !shapesMatch(a.shape, b.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.shape, b.shape))
	}

	// CPU fallback — ensure GPU tensors have CPU data available
	if a.IsOnGPU() {
		if _, err := a.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Sub: failed to get CPU data for a: %v", err))
		}
	}
	if b.IsOnGPU() {
		if _, err := b.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Sub: failed to get CPU data for b: %v", err))
		}
	}

	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)
		// Use SIMD-optimized vector subtraction
		vectorSub(cData, aData, bData)
	default:
		panic(fmt.Sprintf("Sub not implemented for dtype %s", a.dtype))
	}

	return result
}

// Mul performs element-wise multiplication: C = A * B
// Supports GPU acceleration when both tensors are on GPU
func Mul(a, b *Tensor) *Tensor {
	if !shapesMatch(a.shape, b.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.shape, b.shape))
	}

	// Try GPU path first
	if a.IsOnGPU() && b.IsOnGPU() {
		result := mulGPU(a, b)
		if result != nil {
			return result
		}
	}

	// CPU fallback — ensure GPU tensors have CPU data available
	if a.IsOnGPU() {
		if _, err := a.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Mul: failed to get CPU data for a: %v", err))
		}
	}
	if b.IsOnGPU() {
		if _, err := b.EnsureCPUData(); err != nil {
			panic(fmt.Sprintf("Mul: failed to get CPU data for b: %v", err))
		}
	}

	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)
		// Use SIMD-optimized vector multiplication
		vectorMul(cData, aData, bData)
	default:
		panic(fmt.Sprintf("Mul not implemented for dtype %s", a.dtype))
	}

	return result
}

// Div performs element-wise division: C = A / B
func Div(a, b *Tensor) *Tensor {
	if !shapesMatch(a.shape, b.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.shape, b.shape))
	}

	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)
		for i := range aData {
			cData[i] = aData[i] / bData[i]
		}
	default:
		panic(fmt.Sprintf("Div not implemented for dtype %s", a.dtype))
	}

	return result
}

// Scalar Operations

// AddScalar adds a scalar to all elements: C = A + scalar
func AddScalar(a *Tensor, scalar float32) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i := range aData {
			cData[i] = aData[i] + scalar
		}
	default:
		panic(fmt.Sprintf("AddScalar not implemented for dtype %s", a.dtype))
	}

	return result
}

// MulScalar multiplies all elements by a scalar: C = A * scalar
func MulScalar(a *Tensor, scalar float32) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i := range aData {
			cData[i] = aData[i] * scalar
		}
	default:
		panic(fmt.Sprintf("MulScalar not implemented for dtype %s", a.dtype))
	}

	return result
}

// DivScalar divides all elements by a scalar: C = A / scalar
func DivScalar(a *Tensor, scalar float32) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i := range aData {
			cData[i] = aData[i] / scalar
		}
	default:
		panic(fmt.Sprintf("DivScalar not implemented for dtype %s", a.dtype))
	}

	return result
}

// Reduction Operations

// Sum reduces tensor along the last dimension
// TODO: Support specifying dimension
func Sum(a *Tensor) float32 {
	var sum float32

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		for _, val := range aData {
			sum += val
		}
	default:
		panic(fmt.Sprintf("Sum not implemented for dtype %s", a.dtype))
	}

	return sum
}

// Mean computes the average of all elements
func Mean(a *Tensor) float32 {
	return Sum(a) / float32(a.Size())
}

// Max finds the maximum value
func Max(a *Tensor) float32 {
	var maxVal float32

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		if len(aData) == 0 {
			return 0.0
		}
		maxVal = aData[0]
		for _, val := range aData {
			if val > maxVal {
				maxVal = val
			}
		}
	default:
		panic(fmt.Sprintf("Max not implemented for dtype %s", a.dtype))
	}

	return maxVal
}

// Min finds the minimum value
func Min(a *Tensor) float32 {
	var minVal float32

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		if len(aData) == 0 {
			return 0.0
		}
		minVal = aData[0]
		for _, val := range aData {
			if val < minVal {
				minVal = val
			}
		}
	default:
		panic(fmt.Sprintf("Min not implemented for dtype %s", a.dtype))
	}

	return minVal
}

// Activation Functions

// ReLU applies ReLU activation: max(0, x)
func ReLU(a *Tensor) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i, val := range aData {
			if val > 0 {
				cData[i] = val
			} else {
				cData[i] = 0
			}
		}
	default:
		panic(fmt.Sprintf("ReLU not implemented for dtype %s", a.dtype))
	}

	return result
}

// GELU applies GELU activation (Gaussian Error Linear Unit)
func GELU(a *Tensor) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i, x := range aData {
			// GELU(x) ≈ x * Φ(x) where Φ is the cumulative distribution function
			// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
			cdf := float32(0.5) * (1.0 + float32(math.Tanh(float64(
				math.Sqrt(2.0/math.Pi) * (float64(x) + 0.044715*math.Pow(float64(x), 3))))))
			cData[i] = x * cdf
		}
	default:
		panic(fmt.Sprintf("GELU not implemented for dtype %s", a.dtype))
	}

	return result
}

// SiLU applies SiLU (Swish) activation: x * sigmoid(x)
// Supports GPU acceleration when tensor is on GPU
func SiLU(a *Tensor) *Tensor {
	// Try GPU path first
	if a.IsOnGPU() {
		result := siluGPU(a)
		if result != nil {
			return result
		}
	}

	// CPU fallback
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i, x := range aData {
			sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-x))))
			cData[i] = x * sigmoid
		}
	default:
		panic(fmt.Sprintf("SiLU not implemented for dtype %s", a.dtype))
	}

	return result
}

// Sigmoid applies sigmoid activation: 1 / (1 + exp(-x))
func Sigmoid(a *Tensor) *Tensor {
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)
		for i, x := range aData {
			cData[i] = 1.0 / (1.0 + float32(math.Exp(float64(-x))))
		}
	default:
		panic(fmt.Sprintf("Sigmoid not implemented for dtype %s", a.dtype))
	}

	return result
}

// Softmax applies softmax activation along the last dimension
func Softmax(a *Tensor) *Tensor {
	// Try GPU path first
	if a.IsOnGPU() {
		result := softmaxGPU(a)
		if result != nil {
			return result
		}
	}

	// CPU fallback
	result := NewTensor(a.shape, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		cData := result.data.([]float32)

		// Find max for numerical stability
		maxVal := aData[0]
		for _, val := range aData {
			if val > maxVal {
				maxVal = val
			}
		}

		// Compute exp(x - max) and sum
		var sumExp float32
		for i, val := range aData {
			expVal := float32(math.Exp(float64(val - maxVal)))
			cData[i] = expVal
			sumExp += expVal
		}

		// Normalize
		for i := range cData {
			cData[i] /= sumExp
		}
	default:
		panic(fmt.Sprintf("Softmax not implemented for dtype %s", a.dtype))
	}

	return result
}

// Shape Manipulation

// Reshape changes the tensor shape without copying data
func Reshape(a *Tensor, newShape []int) *Tensor {
	// Verify total size matches
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != a.Size() {
		panic(fmt.Sprintf("cannot reshape tensor of size %d to shape %v (size %d)", a.Size(), newShape, newSize))
	}

	return &Tensor{
		data:       a.data,
		shape:      newShape,
		stride:     computeStrides(newShape),
		dtype:      a.dtype,
		device:     a.device,
		offset:     a.offset,
		gpuBuffer:  a.gpuBuffer,
		gpuDevice:  a.gpuDevice,
		gpuKernels: a.gpuKernels,
	}
}

// Transpose swaps two dimensions (only works for 2D tensors for now)
func Transpose(a *Tensor) *Tensor {
	if len(a.shape) != 2 {
		panic("Transpose only implemented for 2D tensors")
	}

	// Create new tensor with transposed shape
	newShape := []int{a.shape[1], a.shape[0]}
	result := NewTensor(newShape, a.dtype)

	// Copy data with transposed indices
	switch a.dtype {
	case Float32:
		for i := 0; i < a.shape[0]; i++ {
			for j := 0; j < a.shape[1]; j++ {
				val := a.At(i, j)
				result.Set(val, j, i)
			}
		}
	default:
		panic(fmt.Sprintf("Transpose not implemented for dtype %s", a.dtype))
	}

	return result
}

// Helper functions

func shapesMatch(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Dequantize converts a quantized tensor to Float32
// Returns a new tensor with Float32 dtype and the same shape
func Dequantize(t *Tensor) *Tensor {
	// If already Float32, return as-is
	if t.dtype == Float32 {
		return t
	}

	// Dispatch to type-specific dequantization
	var result *Tensor
	var err error

	switch t.dtype {
	case Q4_K:
		result, err = DequantizeQ4_KTensor(t)
	case Q5_K:
		result, err = DequantizeQ5_KTensor(t)
	case Q6_K:
		result, err = DequantizeQ6_KTensor(t)
	case Float16:
		// Dequantize float16 to float32
		numElements := 1
		for _, dim := range t.shape {
			numElements *= dim
		}
		f16Data := t.data.([]uint16)
		f32Data := make([]float32, numElements)
		for i, f16 := range f16Data {
			f32Data[i] = float16ToFloat32(f16)
		}
		result = &Tensor{
			data:   f32Data,
			shape:  append([]int{}, t.shape...),
			stride: append([]int{}, t.stride...),
			dtype:  Float32,
			device: CPU,
		}
	default:
		panic(fmt.Sprintf("Dequantize not implemented for dtype %s", t.dtype))
	}

	if err != nil {
		panic(fmt.Sprintf("Dequantization failed: %v", err))
	}

	return result
}
