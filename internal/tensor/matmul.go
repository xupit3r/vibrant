package tensor

import (
	"fmt"
	"runtime"
	"sync"
)

// MatMul performs matrix multiplication: C = A @ B
// A: [M x K], B: [K x N] -> C: [M x N]
func MatMul(a, b *Tensor) *Tensor {
	// Validate dimensions
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic(fmt.Sprintf("MatMul requires 2D tensors, got %dD and %dD", len(a.shape), len(b.shape)))
	}

	M, K := a.shape[0], a.shape[1]
	K2, N := b.shape[0], b.shape[1]

	if K != K2 {
		panic(fmt.Sprintf("MatMul dimension mismatch: [%d x %d] @ [%d x %d]", M, K, K2, N))
	}

	// Dispatch to best implementation based on size
	if M*N*K < 1024 {
		// Small matrices: use naive implementation
		return matmulNaive(a, b)
	} else if M*N*K < 1024*1024 {
		// Medium matrices: use blocked implementation
		return matmulBlocked(a, b)
	} else {
		// Large matrices: use parallel blocked implementation
		return matmulParallel(a, b)
	}
}

// MatVec performs matrix-vector multiplication: c = A @ b
// A: [M x K], b: [K] -> c: [M]
// Optimized for the common case where b is a vector
func MatVec(a, b *Tensor) *Tensor {
	// Validate dimensions
	if len(a.shape) != 2 {
		panic(fmt.Sprintf("MatVec requires 2D matrix, got %dD", len(a.shape)))
	}
	if len(b.shape) != 1 {
		panic(fmt.Sprintf("MatVec requires 1D vector, got %dD", len(b.shape)))
	}

	M, K := a.shape[0], a.shape[1]
	if b.shape[0] != K {
		panic(fmt.Sprintf("MatVec dimension mismatch: [%d x %d] @ [%d]", M, K, b.shape[0]))
	}

	result := NewTensor([]int{M}, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		for i := 0; i < M; i++ {
			sum := float32(0)
			aRow := aData[i*K : (i+1)*K]
			for k := 0; k < K; k++ {
				sum += aRow[k] * bData[k]
			}
			cData[i] = sum
		}
	default:
		panic(fmt.Sprintf("MatVec not implemented for dtype %s", a.dtype))
	}

	return result
}

// matmulNaive implements naive triple-loop matrix multiplication
// Simple and correct, but not optimized
func matmulNaive(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := NewTensor([]int{M, N}, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				sum := float32(0)
				for k := 0; k < K; k++ {
					sum += aData[i*K+k] * bData[k*N+j]
				}
				cData[i*N+j] = sum
			}
		}
	default:
		panic(fmt.Sprintf("matmulNaive not implemented for dtype %s", a.dtype))
	}

	return result
}

// matmulBlocked implements cache-friendly blocked matrix multiplication
// Uses tiling to improve cache locality
func matmulBlocked(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := Zeros([]int{M, N}, a.dtype)

	// Block size tuned for L1 cache (32KB / 4 bytes = 8K floats, sqrt(8K) ≈ 90)
	// Use smaller blocks to fit in cache
	const blockSize = 32

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		// Block the outer loops for cache locality
		for i0 := 0; i0 < M; i0 += blockSize {
			for j0 := 0; j0 < N; j0 += blockSize {
				for k0 := 0; k0 < K; k0 += blockSize {
					// Inner micro-kernel
					iMax := min(i0+blockSize, M)
					jMax := min(j0+blockSize, N)
					kMax := min(k0+blockSize, K)

					for i := i0; i < iMax; i++ {
						for j := j0; j < jMax; j++ {
							sum := float32(0)
							for k := k0; k < kMax; k++ {
								sum += aData[i*K+k] * bData[k*N+j]
							}
							cData[i*N+j] += sum
						}
					}
				}
			}
		}
	default:
		panic(fmt.Sprintf("matmulBlocked not implemented for dtype %s", a.dtype))
	}

	return result
}

// matmulParallel implements parallel matrix multiplication using goroutines
// Divides work across CPU cores
func matmulParallel(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := Zeros([]int{M, N}, a.dtype)

	// Use number of CPUs for parallelism
	numWorkers := runtime.NumCPU()
	rowsPerWorker := (M + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		for worker := 0; worker < numWorkers; worker++ {
			wg.Add(1)
			startRow := worker * rowsPerWorker
			endRow := min(startRow+rowsPerWorker, M)

			go func(start, end int) {
				defer wg.Done()

				// Each worker processes a block of rows
				// Use blocking within each worker's section
				const blockSize = 32

				for i := start; i < end; i++ {
					for j0 := 0; j0 < N; j0 += blockSize {
						jMax := min(j0+blockSize, N)

						for j := j0; j < jMax; j++ {
							sum := float32(0)
							for k := 0; k < K; k++ {
								sum += aData[i*K+k] * bData[k*N+j]
							}
							cData[i*N+j] = sum
						}
					}
				}
			}(startRow, endRow)
		}

		wg.Wait()
	default:
		panic(fmt.Sprintf("matmulParallel not implemented for dtype %s", a.dtype))
	}

	return result
}

// BatchMatMul performs batched matrix multiplication
// A: [B x M x K], B: [B x K x N] -> C: [B x M x N]
func BatchMatMul(a, b *Tensor) *Tensor {
	if len(a.shape) != 3 || len(b.shape) != 3 {
		panic(fmt.Sprintf("BatchMatMul requires 3D tensors, got %dD and %dD", len(a.shape), len(b.shape)))
	}

	batchSize := a.shape[0]
	M, K := a.shape[1], a.shape[2]
	K2, N := b.shape[1], b.shape[2]

	if a.shape[0] != b.shape[0] {
		panic(fmt.Sprintf("BatchMatMul batch size mismatch: %d vs %d", a.shape[0], b.shape[0]))
	}
	if K != K2 {
		panic(fmt.Sprintf("BatchMatMul dimension mismatch: [%d x %d x %d] @ [%d x %d x %d]",
			batchSize, M, K, batchSize, K2, N))
	}

	result := NewTensor([]int{batchSize, M, N}, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		// Process each batch item
		for batch := 0; batch < batchSize; batch++ {
			aOffset := batch * M * K
			bOffset := batch * K * N
			cOffset := batch * M * N

			// Standard matrix multiplication for this batch
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					sum := float32(0)
					for k := 0; k < K; k++ {
						sum += aData[aOffset+i*K+k] * bData[bOffset+k*N+j]
					}
					cData[cOffset+i*N+j] = sum
				}
			}
		}
	default:
		panic(fmt.Sprintf("BatchMatMul not implemented for dtype %s", a.dtype))
	}

	return result
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
