package tensor

import (
	"fmt"
	"runtime"
	"sync"
)

// Phase 3b: Improved block-cached implementation with better memory access pattern
// Instead of random block access, we process B column-wise to maximize cache hits

// MatMulQ5KBlockedV2 - Improved version with column-major B access
func MatMulQ5KBlockedV2(a, b *Tensor) (*Tensor, error) {
	if a.dtype != Float32 {
		return nil, fmt.Errorf("MatMulQ5KBlockedV2: input A must be Float32, got %v", a.dtype)
	}
	if b.dtype != Q5_K {
		return nil, fmt.Errorf("MatMulQ5KBlockedV2: input B must be Q5_K, got %v", b.dtype)
	}
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("MatMulQ5KBlockedV2: both inputs must be 2D")
	}
	if a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("MatMulQ5KBlockedV2: incompatible shapes")
	}

	M, K, N := a.shape[0], a.shape[1], b.shape[1]

	// Small matrices: use single-threaded version
	if M < 32 {
		return matmulQ5KBlockedV2Kernel(a, b, M, K, N)
	}

	// Large matrices: parallel execution
	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}

	c := NewTensor([]int{M, N}, Float32)
	cData := c.data.([]float32)

	rowsPerWorker := (M + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for worker := 0; worker < numWorkers; worker++ {
		startRow := worker * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > M {
			endRow = M
		}
		if startRow >= M {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matmulQ5KBlockedV2KernelRange(a, b, K, N, cData, start, end)
		}(startRow, endRow)
	}

	wg.Wait()
	return c, nil
}

func matmulQ5KBlockedV2Kernel(a, b *Tensor, M, K, N int) (*Tensor, error) {
	c := NewTensor([]int{M, N}, Float32)
	cData := c.data.([]float32)
	matmulQ5KBlockedV2KernelRange(a, b, K, N, cData, 0, M)
	return c, nil
}

// Key optimization: Process column-by-column in B to maximize block reuse
// For each column j in B, dequantize blocks once and reuse across all rows
func matmulQ5KBlockedV2KernelRange(a, b *Tensor, K, N int, cData []float32, startRow, endRow int) {
	aData := a.data.([]float32)
	bData := b.data.([]byte)

	// Pre-allocate a single dequantized block buffer to avoid allocations
	dequantBuffer := make([]float32, Q5K_BLOCK_ELEMENTS)

	// For each output column (maximizes block reuse)
	for j := 0; j < N; j++ {
		// Track which block we last dequantized
		lastBlockIdx := -1

		// For each row in this column
		for i := startRow; i < endRow; i++ {
			sum := float32(0.0)

			// Compute dot product: C[i,j] = sum(A[i,k] * B[k,j])
			for k := 0; k < K; k++ {
				aVal := aData[i*a.shape[1]+k]

				// Find block containing B[k,j]
				bLinearIdx := k*N + j
				blockIdx := bLinearIdx / Q5K_BLOCK_ELEMENTS
				idxInBlock := bLinearIdx % Q5K_BLOCK_ELEMENTS

				// Dequantize block only if it changed
				if blockIdx != lastBlockIdx {
					blockOffset := blockIdx * Q5_K_BLOCK_SIZE
					if blockOffset+Q5_K_BLOCK_SIZE <= len(bData) {
						q5kBlock := parseQ5_KBlock(bData[blockOffset : blockOffset+Q5_K_BLOCK_SIZE])
						dequantizeQ5KBlockBatch(q5kBlock, dequantBuffer)
					}
					lastBlockIdx = blockIdx
				}

				bVal := dequantBuffer[idxInBlock]
				sum += aVal * bVal
			}

			cData[i*N+j] = sum
		}
	}
}

// Similar improvements for Q6K and Q4K

func MatMulQ6KBlockedV2(a, b *Tensor) (*Tensor, error) {
	if a.dtype != Float32 || b.dtype != Q6_K {
		return nil, fmt.Errorf("MatMulQ6KBlockedV2: invalid dtypes")
	}
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("MatMulQ6KBlockedV2: both inputs must be 2D")
	}
	if a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("MatMulQ6KBlockedV2: incompatible shapes")
	}

	M, K, N := a.shape[0], a.shape[1], b.shape[1]

	if M < 32 {
		c := NewTensor([]int{M, N}, Float32)
		cData := c.data.([]float32)
		matmulQ6KBlockedV2KernelRange(a, b, K, N, cData, 0, M)
		return c, nil
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}

	c := NewTensor([]int{M, N}, Float32)
	cData := c.data.([]float32)
	rowsPerWorker := (M + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for worker := 0; worker < numWorkers; worker++ {
		startRow := worker * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > M {
			endRow = M
		}
		if startRow >= M {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matmulQ6KBlockedV2KernelRange(a, b, K, N, cData, start, end)
		}(startRow, endRow)
	}

	wg.Wait()
	return c, nil
}

func matmulQ6KBlockedV2KernelRange(a, b *Tensor, K, N int, cData []float32, startRow, endRow int) {
	aData := a.data.([]float32)
	bData := b.data.([]byte)
	dequantBuffer := make([]float32, Q5K_BLOCK_ELEMENTS)

	for j := 0; j < N; j++ {
		lastBlockIdx := -1

		for i := startRow; i < endRow; i++ {
			sum := float32(0.0)

			for k := 0; k < K; k++ {
				aVal := aData[i*a.shape[1]+k]
				bLinearIdx := k*N + j
				blockIdx := bLinearIdx / Q5K_BLOCK_ELEMENTS
				idxInBlock := bLinearIdx % Q5K_BLOCK_ELEMENTS

				if blockIdx != lastBlockIdx {
					blockOffset := blockIdx * Q6_K_BLOCK_SIZE
					if blockOffset+Q6_K_BLOCK_SIZE <= len(bData) {
						q6kBlock := parseQ6_KBlock(bData[blockOffset : blockOffset+Q6_K_BLOCK_SIZE])
						dequantizeQ6KBlockBatch(q6kBlock, dequantBuffer)
					}
					lastBlockIdx = blockIdx
				}

				bVal := dequantBuffer[idxInBlock]
				sum += aVal * bVal
			}

			cData[i*N+j] = sum
		}
	}
}

func MatMulQ4KBlockedV2(a, b *Tensor) (*Tensor, error) {
	if a.dtype != Float32 || b.dtype != Q4_K {
		return nil, fmt.Errorf("MatMulQ4KBlockedV2: invalid dtypes")
	}
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("MatMulQ4KBlockedV2: both inputs must be 2D")
	}
	if a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("MatMulQ4KBlockedV2: incompatible shapes")
	}

	M, K, N := a.shape[0], a.shape[1], b.shape[1]

	if M < 32 {
		c := NewTensor([]int{M, N}, Float32)
		cData := c.data.([]float32)
		matmulQ4KBlockedV2KernelRange(a, b, K, N, cData, 0, M)
		return c, nil
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}

	c := NewTensor([]int{M, N}, Float32)
	cData := c.data.([]float32)
	rowsPerWorker := (M + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for worker := 0; worker < numWorkers; worker++ {
		startRow := worker * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > M {
			endRow = M
		}
		if startRow >= M {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matmulQ4KBlockedV2KernelRange(a, b, K, N, cData, start, end)
		}(startRow, endRow)
	}

	wg.Wait()
	return c, nil
}

func matmulQ4KBlockedV2KernelRange(a, b *Tensor, K, N int, cData []float32, startRow, endRow int) {
	aData := a.data.([]float32)
	bData := b.data.([]byte)
	dequantBuffer := make([]float32, Q5K_BLOCK_ELEMENTS)

	for j := 0; j < N; j++ {
		lastBlockIdx := -1

		for i := startRow; i < endRow; i++ {
			sum := float32(0.0)

			for k := 0; k < K; k++ {
				aVal := aData[i*a.shape[1]+k]
				bLinearIdx := k*N + j
				blockIdx := bLinearIdx / Q5K_BLOCK_ELEMENTS
				idxInBlock := bLinearIdx % Q5K_BLOCK_ELEMENTS

				if blockIdx != lastBlockIdx {
					blockOffset := blockIdx * Q4_K_BLOCK_SIZE
					if blockOffset+Q4_K_BLOCK_SIZE <= len(bData) {
						q4kBlock := parseQ4_KBlock(bData[blockOffset : blockOffset+Q4_K_BLOCK_SIZE])
						dequantizeQ4KBlockBatch(q4kBlock, dequantBuffer)
					}
					lastBlockIdx = blockIdx
				}

				bVal := dequantBuffer[idxInBlock]
				sum += aVal * bVal
			}

			cData[i*N+j] = sum
		}
	}
}
