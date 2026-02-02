package tensor

import (
	"runtime"
	"sync"
)

// matmulOuterProduct computes C[M,N] = A[M,K] @ B[K,N] without transposing B.
// Uses outer-product (SAXPY) accumulation: for each k, C[i,:] += A[i,k] * B[k,:]
// B is read row-by-row (cache-friendly even for row-major layout).
func matmulOuterProduct(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]
	result := Zeros([]int{M, N}, Float32)

	aData := a.data.([]float32)
	bData := b.data.([]float32)
	cData := result.data.([]float32)

	for k := 0; k < K; k++ {
		bRow := bData[k*N : (k+1)*N]
		for i := 0; i < M; i++ {
			aVal := aData[i*K+k]
			if aVal == 0 {
				continue
			}
			cRow := cData[i*N : (i+1)*N]
			vectorScaleAdd(cRow, bRow, aVal)
		}
	}
	return result
}

// matmulOuterProductParallel is the parallel version of outer-product matmul.
// Parallelizes over rows of A (each worker owns a block of output rows).
func matmulOuterProductParallel(a, b *Tensor) *Tensor {
	M, K := a.shape[0], a.shape[1]
	N := b.shape[1]
	result := Zeros([]int{M, N}, Float32)

	aData := a.data.([]float32)
	bData := b.data.([]float32)
	cData := result.data.([]float32)

	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}
	rowsPerWorker := (M + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > M {
			endRow = M
		}
		go func(start, end int) {
			defer wg.Done()
			for k := 0; k < K; k++ {
				bRow := bData[k*N : (k+1)*N]
				for i := start; i < end; i++ {
					aVal := aData[i*K+k]
					if aVal == 0 {
						continue
					}
					cRow := cData[i*N : (i+1)*N]
					vectorScaleAdd(cRow, bRow, aVal)
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
	return result
}
