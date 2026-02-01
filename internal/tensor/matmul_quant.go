package tensor

import (
	"fmt"
)

// MatMulQ5K performs fused dequantization + matrix multiplication for Q5_K quantized tensors.
//
// This function eliminates the need to create a full intermediate Float32 tensor by
// dequantizing Q5_K values on-demand during matrix multiplication.
//
// Parameters:
//   - a: Float32 tensor of shape [M, K] (input activations)
//   - bQuantized: Q5_K quantized tensor of shape [K, N] (weights)
//
// Returns:
//   - Float32 tensor of shape [M, N] (output)
//
// Memory savings:
//   - Current approach: Allocates K*N*4 bytes for dequantized tensor
//   - Fused approach: Zero intermediate allocations
//
// Performance:
//   - This is the reference implementation (correctness over speed)
//   - Optimized versions will use blocking, SIMD, and parallelization
//
// Numerical accuracy:
//   - Produces identical results to DequantizeQ5_KTensor() + MatMul()
//   - Uses the same dequantization formula per element
func MatMulQ5K(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
	// Validate inputs
	if a == nil || bQuantized == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}

	if a.DType() != Float32 {
		return nil, fmt.Errorf("input tensor a must be Float32, got %s", a.DType())
	}

	if bQuantized.DType() != Q5_K {
		return nil, fmt.Errorf("input tensor b must be Q5_K, got %s", bQuantized.DType())
	}

	// Check dimensions
	if len(a.shape) != 2 {
		return nil, fmt.Errorf("input tensor a must be 2D, got %dD", len(a.shape))
	}

	if len(bQuantized.shape) != 2 {
		return nil, fmt.Errorf("input tensor b must be 2D, got %dD", len(bQuantized.shape))
	}

	M := a.shape[0]
	K := a.shape[1]
	K2 := bQuantized.shape[0]
	N := bQuantized.shape[1]

	// Validate matrix multiplication dimensions
	if K != K2 {
		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: a[%d,%d] × b[%d,%d]", M, K, K2, N)
	}

	// Create output tensor
	output := NewTensor([]int{M, N}, Float32)
	outputData := output.data.([]float32)

	// Get input data
	aData := a.data.([]float32)
	bData := bQuantized.data.([]byte)

	// Perform fused dequantization + matrix multiplication
	// C[i,j] = sum_k(A[i,k] * B[k,j])
	//
	// For each output element, we compute the dot product of:
	//   - Row i of A (Float32)
	//   - Column j of B (Q5_K, dequantized on-the-fly)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)

			// Dot product: a[i,:] · b[:,j]
			for k := 0; k < K; k++ {
				// Get value from A (already Float32)
				aVal := aData[i*K+k]

				// Dequantize single element from B on-demand
				// B is stored in row-major order: b[k,j] = b[k*N + j]
				bIdx := k*N + j
				bVal := DequantizeQ5_KElement(bData, bIdx)

				sum += aVal * bVal
			}

			// Store result
			outputData[i*N+j] = sum
		}
	}

	return output, nil
}

// MatMulQ6K performs fused dequantization + matrix multiplication for Q6_K quantized tensors.
//
// This function eliminates the need to create a full intermediate Float32 tensor by
// dequantizing Q6_K values on-demand during matrix multiplication.
//
// Parameters:
//   - a: Float32 tensor of shape [M, K] (input activations)
//   - bQuantized: Q6_K quantized tensor of shape [K, N] (weights)
//
// Returns:
//   - Float32 tensor of shape [M, N] (output)
//
// Memory savings:
//   - Current approach: Allocates K*N*4 bytes for dequantized tensor
//   - Fused approach: Zero intermediate allocations
//
// Performance:
//   - This is the reference implementation (correctness over speed)
//   - Optimized versions will use blocking, SIMD, and parallelization
//
// Numerical accuracy:
//   - Produces identical results to DequantizeQ6_KTensor() + MatMul()
//   - Uses the same dequantization formula per element
func MatMulQ6K(a *Tensor, bQuantized *Tensor) (*Tensor, error) {
	// Validate inputs
	if a == nil || bQuantized == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}

	if a.DType() != Float32 {
		return nil, fmt.Errorf("input tensor a must be Float32, got %s", a.DType())
	}

	if bQuantized.DType() != Q6_K {
		return nil, fmt.Errorf("input tensor b must be Q6_K, got %s", bQuantized.DType())
	}

	// Check dimensions
	if len(a.shape) != 2 {
		return nil, fmt.Errorf("input tensor a must be 2D, got %dD", len(a.shape))
	}

	if len(bQuantized.shape) != 2 {
		return nil, fmt.Errorf("input tensor b must be 2D, got %dD", len(bQuantized.shape))
	}

	M := a.shape[0]
	K := a.shape[1]
	K2 := bQuantized.shape[0]
	N := bQuantized.shape[1]

	// Validate matrix multiplication dimensions
	if K != K2 {
		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: a[%d,%d] × b[%d,%d]", M, K, K2, N)
	}

	// Create output tensor
	output := NewTensor([]int{M, N}, Float32)
	outputData := output.data.([]float32)

	// Get input data
	aData := a.data.([]float32)
	bData := bQuantized.data.([]byte)

	// Perform fused dequantization + matrix multiplication
	// C[i,j] = sum_k(A[i,k] * B[k,j])
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)

			// Dot product: a[i,:] · b[:,j]
			for k := 0; k < K; k++ {
				// Get value from A (already Float32)
				aVal := aData[i*K+k]

				// Dequantize single element from B on-demand
				// B is stored in row-major order: b[k,j] = b[k*N + j]
				bIdx := k*N + j
				bVal := DequantizeQ6_KElement(bData, bIdx)

				sum += aVal * bVal
			}

			// Store result
			outputData[i*N+j] = sum
		}
	}

	return output, nil
}

// matmulQuantValidateDimensions validates matrix multiplication dimensions
func matmulQuantValidateDimensions(a, b *Tensor) error {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return fmt.Errorf("both tensors must be 2D")
	}

	if a.shape[1] != b.shape[0] {
		return fmt.Errorf("incompatible dimensions: a[%d,%d] × b[%d,%d]",
			a.shape[0], a.shape[1], b.shape[0], b.shape[1])
	}

	return nil
}
