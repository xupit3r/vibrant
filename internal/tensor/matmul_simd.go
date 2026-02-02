package tensor

import (
	"runtime"
	"sync"
)

// SIMD-optimized matrix multiplication
// Uses vectorized operations for better performance

// matmulSIMD performs matrix multiplication using SIMD-optimized operations
// This version uses better memory access patterns and vectorization hints
func matmulSIMD(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := Zeros([]int{M, N}, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		// Check if B is already pre-transposed (optimization for weight matrices)
		var bTransposed []float32
		if b.IsTransposed() {
			// B is already in [N, K] format (transposed), use it directly
			bTransposed = bData
		} else {
			// Transpose B for better cache locality
			bTransposed = make([]float32, K*N)
			for k := 0; k < K; k++ {
				for j := 0; j < N; j++ {
					bTransposed[j*K+k] = bData[k*N+j]
				}
			}
		}

		// Now compute C = A @ B using transposed B
		// This allows row-by-row dot products (much faster!)
		for i := 0; i < M; i++ {
			aRow := aData[i*K : (i+1)*K]
			cRow := cData[i*N : (i+1)*N]

			for j := 0; j < N; j++ {
				bRow := bTransposed[j*K : (j+1)*K]
				cRow[j] = vectorDotProduct(aRow, bRow)
			}
		}

	default:
		panic("matmulSIMD not implemented for this dtype")
	}

	return result
}

// matmulSIMDBlocked combines SIMD with cache-friendly blocking
func matmulSIMDBlocked(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := Zeros([]int{M, N}, a.dtype)

	const blockSize = 64 // Larger blocks since we have SIMD

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		// Check if B is pre-transposed for optimized access
		if b.IsTransposed() {
			// B is already in [N, K] format - use row-wise access (cache friendly!)
			for i0 := 0; i0 < M; i0 += blockSize {
				for j0 := 0; j0 < N; j0 += blockSize {
					for k0 := 0; k0 < K; k0 += blockSize {
						iMax := min(i0+blockSize, M)
						jMax := min(j0+blockSize, N)
						kMax := min(k0+blockSize, K)

						for i := i0; i < iMax; i++ {
							for j := j0; j < jMax; j++ {
								sum := float32(0)
								// B is transposed: access row j (which is column j of original B)
								for k := k0; k < kMax; k++ {
									sum += aData[i*K+k] * bData[j*K+k]
								}
								cData[i*N+j] += sum
							}
						}
					}
				}
			}
		} else {
			// B is in [K, N] format - use column-wise access (original code path)
			for i0 := 0; i0 < M; i0 += blockSize {
				for j0 := 0; j0 < N; j0 += blockSize {
					for k0 := 0; k0 < K; k0 += blockSize {
						// Inner micro-kernel with SIMD
						iMax := min(i0+blockSize, M)
						jMax := min(j0+blockSize, N)
						kMax := min(k0+blockSize, K)

						for i := i0; i < iMax; i++ {
							for j := j0; j < jMax; j++ {
								// Vectorized accumulation
								sum := float32(0)

								// Extract column j from block
								for k := k0; k < kMax; k++ {
									sum += aData[i*K+k] * bData[k*N+j]
								}

								cData[i*N+j] += sum
							}
						}
					}
				}
			}
		}

	default:
		panic("matmulSIMDBlocked not implemented for this dtype")
	}

	return result
}

// matmulSIMDSingleRow handles the M=1 case (decode step).
// Parallelizes over output columns instead of rows.
func matmulSIMDSingleRow(a, b *Tensor) *Tensor {
	K := a.shape[1]
	N := b.shape[1]
	result := Zeros([]int{1, N}, Float32)

	aData := a.data.([]float32)
	bData := b.data.([]float32)
	cData := result.data.([]float32)
	aRow := aData[:K]

	// If B is pre-transposed, rows of B^T are contiguous — just dot product
	if b.IsTransposed() {
		numWorkers := runtime.NumCPU()
		if numWorkers > N {
			numWorkers = N
		}
		colsPerWorker := (N + numWorkers - 1) / numWorkers

		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			startCol := w * colsPerWorker
			endCol := startCol + colsPerWorker
			if endCol > N {
				endCol = N
			}
			go func(start, end int) {
				defer wg.Done()
				for j := start; j < end; j++ {
					bRow := bData[j*K : (j+1)*K]
					cData[j] = vectorDotProduct(aRow, bRow)
				}
			}(startCol, endCol)
		}
		wg.Wait()
	} else {
		// B not transposed — use outer product for single row
		for k := 0; k < K; k++ {
			aVal := aRow[k]
			if aVal == 0 {
				continue
			}
			bRow := bData[k*N : (k+1)*N]
			vectorScaleAdd(cData, bRow, aVal)
		}
	}
	return result
}

// matmulSIMDParallel combines SIMD with parallel execution
func matmulSIMDParallel(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]

	result := Zeros([]int{M, N}, a.dtype)

	switch a.dtype {
	case Float32:
		aData := a.data.([]float32)
		bData := b.data.([]float32)
		cData := result.data.([]float32)

		var bTransposed []float32

		// Check if B is already transposed (pre-transposed weight matrix)
		if b.IsTransposed() {
			// B is already in transposed form [N, K] - use directly!
			bTransposed = bData
		} else {
			// Transpose B for better cache locality (runtime cost)
			bTransposed = make([]float32, K*N)
			for k := 0; k < K; k++ {
				for j := 0; j < N; j++ {
					bTransposed[j*K+k] = bData[k*N+j]
				}
			}
		}

		// Parallel computation with SIMD dot products
		numWorkers := runtime.NumCPU()
		rowsPerWorker := (M + numWorkers - 1) / numWorkers

		var wg sync.WaitGroup
		for worker := 0; worker < numWorkers; worker++ {
			wg.Add(1)
			startRow := worker * rowsPerWorker
			endRow := min(startRow+rowsPerWorker, M)

			go func(start, end int) {
				defer wg.Done()

				for i := start; i < end; i++ {
					aRow := aData[i*K : (i+1)*K]
					cRow := cData[i*N : (i+1)*N]

					for j := 0; j < N; j++ {
						bRow := bTransposed[j*K : (j+1)*K]
						cRow[j] = vectorDotProduct(aRow, bRow)
					}
				}
			}(startRow, endRow)
		}

		wg.Wait()

	default:
		panic("matmulSIMDParallel not implemented for this dtype")
	}

	return result
}
