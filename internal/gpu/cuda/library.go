// +build linux,cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lstdc++

#include <cuda_runtime.h>
#include <stdlib.h>

// External kernel declarations (defined in kernels.cu, linked separately)
extern void matmul_f32_launch(
	const float* A, const float* B, float* C,
	int M, int N, int K,
	cudaStream_t stream
);

extern void matmul_f32_single_row_launch(
	const float* A, const float* B, float* C,
	int N, int K,
	cudaStream_t stream
);

extern void softmax_f32_launch(
	const float* input, float* output,
	int size,
	cudaStream_t stream
);

extern void softmax_batched_f32_launch(
	const float* input, float* output,
	int batch_size, int size,
	cudaStream_t stream
);

extern void rms_norm_f32_launch(
	const float* input, const float* weight, float* output,
	int size, float eps,
	cudaStream_t stream
);

extern void rms_norm_batched_f32_launch(
	const float* input, const float* weight, float* output,
	int batch_size, int size, float eps,
	cudaStream_t stream
);

extern void add_f32_launch(
	const float* A, const float* B, float* C,
	int size,
	cudaStream_t stream
);

extern void mul_f32_launch(
	const float* A, const float* B, float* C,
	int size,
	cudaStream_t stream
);

extern void mul_scalar_f32_launch(
	const float* A, float scalar, float* B,
	int size,
	cudaStream_t stream
);

extern void silu_f32_launch(
	const float* input, float* output,
	int size,
	cudaStream_t stream
);

extern void copy_f32_launch(
	const float* src, float* dst,
	int size,
	cudaStream_t stream
);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// LaunchMatMulKernel calls the pre-compiled matmul kernel
func LaunchMatMulKernel(A, B, C unsafe.Pointer, M, N, K int, stream unsafe.Pointer) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("null pointer passed to matmul kernel")
	}
	if M <=0 || N <= 0 || K <= 0 {
		return fmt.Errorf("invalid dimensions: M=%d, N=%d, K=%d", M, N, K)
	}

	// Cast stream to CUDA stream type
	cudaStream := (C.cudaStream_t)(stream)
	
	// Launch kernel via C wrapper
	C.matmul_f32_launch(
		(*C.float)(A),
		(*C.float)(B),
		(*C.float)(C),
		C.int(M),
		C.int(N),
		C.int(K),
		cudaStream,
	)
	
	// Check for launch errors
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("matmul kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchMatMulSingleRowKernel calls the optimized single-row matmul kernel
func LaunchMatMulSingleRowKernel(A, B, C unsafe.Pointer, N, K int, stream unsafe.Pointer) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("null pointer passed to matmul_single_row kernel")
	}
	if N <= 0 || K <= 0 {
		return fmt.Errorf("invalid dimensions: N=%d, K=%d", N, K)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.matmul_f32_single_row_launch(
		(*C.float)(A),
		(*C.float)(B),
		(*C.float)(C),
		C.int(N),
		C.int(K),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("matmul_single_row kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchSoftmaxKernel calls the softmax kernel
func LaunchSoftmaxKernel(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if input == nil || output == nil {
		return fmt.Errorf("null pointer passed to softmax kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.softmax_f32_launch(
		(*C.float)(input),
		(*C.float)(output),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("softmax kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchSoftmaxBatchedKernel calls the batched softmax kernel
func LaunchSoftmaxBatchedKernel(input, output unsafe.Pointer, batchSize, size int, stream unsafe.Pointer) error {
	if input == nil || output == nil {
		return fmt.Errorf("null pointer passed to softmax_batched kernel")
	}
	if batchSize <= 0 || size <= 0 {
		return fmt.Errorf("invalid dimensions: batchSize=%d, size=%d", batchSize, size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.softmax_batched_f32_launch(
		(*C.float)(input),
		(*C.float)(output),
		C.int(batchSize),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("softmax_batched kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchRMSNormKernel calls the RMS normalization kernel
func LaunchRMSNormKernel(input, weight, output unsafe.Pointer, size int, eps float32, stream unsafe.Pointer) error {
	if input == nil || weight == nil || output == nil {
		return fmt.Errorf("null pointer passed to rms_norm kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.rms_norm_f32_launch(
		(*C.float)(input),
		(*C.float)(weight),
		(*C.float)(output),
		C.int(size),
		C.float(eps),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("rms_norm kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchRMSNormBatchedKernel calls the batched RMS normalization kernel
func LaunchRMSNormBatchedKernel(input, weight, output unsafe.Pointer, batchSize, size int, eps float32, stream unsafe.Pointer) error {
	if input == nil || weight == nil || output == nil {
		return fmt.Errorf("null pointer passed to rms_norm_batched kernel")
	}
	if batchSize <= 0 || size <= 0 {
		return fmt.Errorf("invalid dimensions: batchSize=%d, size=%d", batchSize, size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.rms_norm_batched_f32_launch(
		(*C.float)(input),
		(*C.float)(weight),
		(*C.float)(output),
		C.int(batchSize),
		C.int(size),
		C.float(eps),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("rms_norm_batched kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchAddKernel calls the element-wise addition kernel
func LaunchAddKernel(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("null pointer passed to add kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.add_f32_launch(
		(*C.float)(A),
		(*C.float)(B),
		(*C.float)(C),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("add kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchMulKernel calls the element-wise multiplication kernel
func LaunchMulKernel(A, B, C unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("null pointer passed to mul kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.mul_f32_launch(
		(*C.float)(A),
		(*C.float)(B),
		(*C.float)(C),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("mul kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchMulScalarKernel calls the scalar multiplication kernel
func LaunchMulScalarKernel(A unsafe.Pointer, scalar float32, B unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if A == nil || B == nil {
		return fmt.Errorf("null pointer passed to mul_scalar kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.mul_scalar_f32_launch(
		(*C.float)(A),
		C.float(scalar),
		(*C.float)(B),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("mul_scalar kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchSiLUKernel calls the SiLU activation kernel
func LaunchSiLUKernel(input, output unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if input == nil || output == nil {
		return fmt.Errorf("null pointer passed to silu kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.silu_f32_launch(
		(*C.float)(input),
		(*C.float)(output),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("silu kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}

// LaunchCopyKernel calls the copy kernel
func LaunchCopyKernel(src, dst unsafe.Pointer, size int, stream unsafe.Pointer) error {
	if src == nil || dst == nil {
		return fmt.Errorf("null pointer passed to copy kernel")
	}
	if size <= 0 {
		return fmt.Errorf("invalid size: %d", size)
	}

	cudaStream := (C.cudaStream_t)(stream)
	
	C.copy_f32_launch(
		(*C.float)(src),
		(*C.float)(dst),
		C.int(size),
		cudaStream,
	)
	
	err := C.cudaGetLastError()
	if err != C.cudaSuccess {
		return fmt.Errorf("copy kernel launch failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	
	return nil
}
