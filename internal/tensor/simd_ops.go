package tensor

// SIMD-optimized operations
// These functions are written to be auto-vectorized by the Go compiler
// The compiler can generate AVX2/NEON instructions when appropriate

// vectorAdd performs element-wise addition with SIMD optimization
func vectorAdd(dst, a, b []float32) {
	// The Go compiler can auto-vectorize this loop with AVX2/NEON
	// Key: contiguous memory access, no dependencies between iterations
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// vectorMul performs element-wise multiplication with SIMD optimization
func vectorMul(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

// vectorSub performs element-wise subtraction with SIMD optimization
func vectorSub(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// vectorDiv performs element-wise division with SIMD optimization
func vectorDiv(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

// vectorAddScalar adds a scalar to all elements with SIMD optimization
func vectorAddScalar(dst, a []float32, scalar float32) {
	for i := range dst {
		dst[i] = a[i] + scalar
	}
}

// vectorMulScalar multiplies all elements by a scalar with SIMD optimization
func vectorMulScalar(dst, a []float32, scalar float32) {
	for i := range dst {
		dst[i] = a[i] * scalar
	}
}

// vectorDivScalar divides all elements by a scalar with SIMD optimization
func vectorDivScalar(dst, a []float32, scalar float32) {
	for i := range dst {
		dst[i] = a[i] / scalar
	}
}

// vectorDotProduct computes dot product with SIMD optimization
func vectorDotProduct(a, b []float32) float32 {
	// Accumulate in 4 separate variables to help with SIMD
	// This allows parallel execution of 4 independent accumulations
	sum0, sum1, sum2, sum3 := float32(0), float32(0), float32(0), float32(0)

	// Process 4 elements at a time
	i := 0
	for ; i+3 < len(a); i += 4 {
		sum0 += a[i+0] * b[i+0]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		sum0 += a[i] * b[i]
	}

	return sum0 + sum1 + sum2 + sum3
}

// vectorSum computes sum of all elements with SIMD optimization
func vectorSum(a []float32) float32 {
	// Use 4-way accumulation for SIMD
	sum0, sum1, sum2, sum3 := float32(0), float32(0), float32(0), float32(0)

	i := 0
	for ; i+3 < len(a); i += 4 {
		sum0 += a[i+0]
		sum1 += a[i+1]
		sum2 += a[i+2]
		sum3 += a[i+3]
	}

	for ; i < len(a); i++ {
		sum0 += a[i]
	}

	return sum0 + sum1 + sum2 + sum3
}

// vectorMax finds maximum element with SIMD optimization
func vectorMax(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}

	// Use 4-way comparison for SIMD
	max0, max1, max2, max3 := a[0], a[0], a[0], a[0]

	i := 0
	for ; i+3 < len(a); i += 4 {
		if a[i+0] > max0 {
			max0 = a[i+0]
		}
		if a[i+1] > max1 {
			max1 = a[i+1]
		}
		if a[i+2] > max2 {
			max2 = a[i+2]
		}
		if a[i+3] > max3 {
			max3 = a[i+3]
		}
	}

	for ; i < len(a); i++ {
		if a[i] > max0 {
			max0 = a[i]
		}
	}

	// Reduce 4 maximums to 1
	if max1 > max0 {
		max0 = max1
	}
	if max2 > max0 {
		max0 = max2
	}
	if max3 > max0 {
		max0 = max3
	}

	return max0
}

// vectorMin finds minimum element with SIMD optimization
func vectorMin(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}

	// Use 4-way comparison for SIMD
	min0, min1, min2, min3 := a[0], a[0], a[0], a[0]

	i := 0
	for ; i+3 < len(a); i += 4 {
		if a[i+0] < min0 {
			min0 = a[i+0]
		}
		if a[i+1] < min1 {
			min1 = a[i+1]
		}
		if a[i+2] < min2 {
			min2 = a[i+2]
		}
		if a[i+3] < min3 {
			min3 = a[i+3]
		}
	}

	for ; i < len(a); i++ {
		if a[i] < min0 {
			min0 = a[i]
		}
	}

	// Reduce 4 minimums to 1
	if min1 < min0 {
		min0 = min1
	}
	if min2 < min0 {
		min0 = min2
	}
	if min3 < min0 {
		min0 = min3
	}

	return min0
}

// vectorReLU applies ReLU activation with SIMD optimization
func vectorReLU(dst, a []float32) {
	for i := range dst {
		if a[i] > 0 {
			dst[i] = a[i]
		} else {
			dst[i] = 0
		}
	}
}

// vectorCopy copies data with SIMD optimization
func vectorCopy(dst, src []float32) {
	// Built-in copy is already optimized, but we keep this for consistency
	copy(dst, src)
}

// vectorScale scales all elements by a factor with SIMD optimization
func vectorScale(dst, a []float32, scale float32) {
	for i := range dst {
		dst[i] = a[i] * scale
	}
}

// vectorFMA performs fused multiply-add: dst = a*b + c with SIMD optimization
func vectorFMA(dst, a, b, c []float32) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}
