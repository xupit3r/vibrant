// +build darwin,cgo

package metal

import (
	"fmt"
	"unsafe"
)

// KernelSet contains all compiled kernels
type KernelSet struct {
	library *Library
	
	// Matrix operations
	MatMul          *Pipeline
	MatMulSingleRow *Pipeline
	
	// Normalization
	Softmax        *Pipeline
	SoftmaxBatched *Pipeline
	RMSNorm        *Pipeline
	RMSNormBatched *Pipeline
	
	// Element-wise operations
	Add       *Pipeline
	Mul       *Pipeline
	MulScalar *Pipeline
	SiLU      *Pipeline
	Copy      *Pipeline
}

// NewKernelSet compiles all kernels
func NewKernelSet(devicePtr unsafe.Pointer) (*KernelSet, error) {
	lib, err := CompileLibrary(devicePtr)
	if err != nil {
		return nil, fmt.Errorf("failed to compile library: %w", err)
	}

	ks := &KernelSet{library: lib}

	// Compile all kernels
	kernels := []struct {
		name     string
		pipeline **Pipeline
	}{
		{"matmul_f32", &ks.MatMul},
		{"matmul_f32_single_row", &ks.MatMulSingleRow},
		{"softmax_f32", &ks.Softmax},
		{"softmax_batched_f32", &ks.SoftmaxBatched},
		{"rms_norm_f32", &ks.RMSNorm},
		{"rms_norm_batched_f32", &ks.RMSNormBatched},
		{"add_f32", &ks.Add},
		{"mul_f32", &ks.Mul},
		{"mul_scalar_f32", &ks.MulScalar},
		{"silu_f32", &ks.SiLU},
		{"copy_f32", &ks.Copy},
	}

	for _, k := range kernels {
		pipeline, err := lib.CreatePipeline(k.name)
		if err != nil {
			ks.Free()
			return nil, fmt.Errorf("failed to create pipeline '%s': %w", k.name, err)
		}
		*k.pipeline = pipeline
	}

	return ks, nil
}

// Free releases all kernels
func (ks *KernelSet) Free() {
	if ks.MatMul != nil {
		ks.MatMul.Free()
	}
	if ks.MatMulSingleRow != nil {
		ks.MatMulSingleRow.Free()
	}
	if ks.Softmax != nil {
		ks.Softmax.Free()
	}
	if ks.SoftmaxBatched != nil {
		ks.SoftmaxBatched.Free()
	}
	if ks.RMSNorm != nil {
		ks.RMSNorm.Free()
	}
	if ks.RMSNormBatched != nil {
		ks.RMSNormBatched.Free()
	}
	if ks.Add != nil {
		ks.Add.Free()
	}
	if ks.Mul != nil {
		ks.Mul.Free()
	}
	if ks.MulScalar != nil {
		ks.MulScalar.Free()
	}
	if ks.SiLU != nil {
		ks.SiLU.Free()
	}
	if ks.Copy != nil {
		ks.Copy.Free()
	}
	if ks.library != nil {
		ks.library.Free()
	}
}

// MatMulParams contains parameters for matrix multiplication
type MatMulParams struct {
	A          unsafe.Pointer // Input matrix A [M x K]
	B          unsafe.Pointer // Input matrix B [K x N]
	C          unsafe.Pointer // Output matrix C [M x N]
	M, N, K    uint32         // Matrix dimensions
	SingleRow  bool           // Use optimized single-row kernel (M=1)
}

// DispatchMatMul executes matrix multiplication on GPU
func (ks *KernelSet) DispatchMatMul(queuePtr unsafe.Pointer, params MatMulParams) error {
	if params.M == 0 || params.N == 0 || params.K == 0 {
		return fmt.Errorf("invalid matrix dimensions: M=%d, N=%d, K=%d", params.M, params.N, params.K)
	}

	// Choose kernel based on whether it's a single-row multiplication
	if params.SingleRow || params.M == 1 {
		// Single-row optimization for decode step
		return ks.MatMulSingleRow.Dispatch(queuePtr, DispatchParams{
			Buffers:     []unsafe.Pointer{params.A, params.B, params.C, unsafe.Pointer(&params.N), unsafe.Pointer(&params.K)},
			BufferSizes: []uint64{0, 0, 0, 4, 4}, // Buffers, then uint32 values
			ThreadsX:    uint(params.N),
			ThreadsY:    1,
			ThreadsZ:    1,
		})
	}

	// General matrix multiplication
	return ks.MatMul.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, params.C, unsafe.Pointer(&params.M), unsafe.Pointer(&params.N), unsafe.Pointer(&params.K)},
		BufferSizes: []uint64{0, 0, 0, 4, 4, 4}, // Buffers, then uint32 values
		ThreadsX:    uint(params.N),
		ThreadsY:    uint(params.M),
		ThreadsZ:    1,
	})
}

// SoftmaxParams contains parameters for softmax
type SoftmaxParams struct {
	Input  unsafe.Pointer // Input tensor
	Output unsafe.Pointer // Output tensor
	Size   uint32         // Size of input/output
}

// DispatchSoftmax executes softmax on GPU
func (ks *KernelSet) DispatchSoftmax(queuePtr unsafe.Pointer, params SoftmaxParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid softmax size: %d", params.Size)
	}

	return ks.Softmax.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.Input, params.Output, unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// SoftmaxBatchedParams contains parameters for batched softmax
type SoftmaxBatchedParams struct {
	Input     unsafe.Pointer // Input tensor [batch_size x seq_len]
	Output    unsafe.Pointer // Output tensor [batch_size x seq_len]
	BatchSize uint32         // Number of sequences
	SeqLen    uint32         // Length of each sequence
}

// DispatchSoftmaxBatched executes batched softmax on GPU
func (ks *KernelSet) DispatchSoftmaxBatched(queuePtr unsafe.Pointer, params SoftmaxBatchedParams) error {
	if params.BatchSize == 0 || params.SeqLen == 0 {
		return fmt.Errorf("invalid softmax dimensions: batch_size=%d, seq_len=%d", params.BatchSize, params.SeqLen)
	}

	return ks.SoftmaxBatched.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.Input, params.Output, unsafe.Pointer(&params.BatchSize), unsafe.Pointer(&params.SeqLen)},
		BufferSizes: []uint64{0, 0, 4, 4},
		ThreadsX:    uint(params.SeqLen),
		ThreadsY:    uint(params.BatchSize),
		ThreadsZ:    1,
	})
}

// RMSNormParams contains parameters for RMS normalization
type RMSNormParams struct {
	Input   unsafe.Pointer // Input tensor
	Weight  unsafe.Pointer // Weight tensor (same size as input)
	Output  unsafe.Pointer // Output tensor
	Size    uint32         // Size of tensors
	Epsilon float32        // Small constant for numerical stability
}

// DispatchRMSNorm executes RMS normalization on GPU
func (ks *KernelSet) DispatchRMSNorm(queuePtr unsafe.Pointer, params RMSNormParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid RMSNorm size: %d", params.Size)
	}

	return ks.RMSNorm.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.Input, params.Weight, params.Output, unsafe.Pointer(&params.Size), unsafe.Pointer(&params.Epsilon)},
		BufferSizes: []uint64{0, 0, 0, 4, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// RMSNormBatchedParams contains parameters for batched RMS normalization
type RMSNormBatchedParams struct {
	Input     unsafe.Pointer // Input tensor [batch_size x dim]
	Weight    unsafe.Pointer // Weight tensor [dim]
	Output    unsafe.Pointer // Output tensor [batch_size x dim]
	BatchSize uint32         // Number of sequences
	Dim       uint32         // Dimension of each sequence
	Epsilon   float32        // Small constant for numerical stability
}

// DispatchRMSNormBatched executes batched RMS normalization on GPU
func (ks *KernelSet) DispatchRMSNormBatched(queuePtr unsafe.Pointer, params RMSNormBatchedParams) error {
	if params.BatchSize == 0 || params.Dim == 0 {
		return fmt.Errorf("invalid RMSNorm dimensions: batch_size=%d, dim=%d", params.BatchSize, params.Dim)
	}

	return ks.RMSNormBatched.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.Input, params.Weight, params.Output, unsafe.Pointer(&params.BatchSize), unsafe.Pointer(&params.Dim), unsafe.Pointer(&params.Epsilon)},
		BufferSizes: []uint64{0, 0, 0, 4, 4, 4},
		ThreadsX:    uint(params.Dim),
		ThreadsY:    uint(params.BatchSize),
		ThreadsZ:    1,
	})
}

// ElementwiseParams contains parameters for element-wise operations
type ElementwiseParams struct {
	A      unsafe.Pointer // Input tensor A
	B      unsafe.Pointer // Input tensor B (or output for unary ops)
	C      unsafe.Pointer // Output tensor C (for binary ops)
	Size   uint32         // Number of elements
	Scalar float32        // Scalar value (for scalar ops)
}

// DispatchAdd executes element-wise addition: C = A + B
func (ks *KernelSet) DispatchAdd(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid size: %d", params.Size)
	}

	return ks.Add.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, params.C, unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 0, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// DispatchMul executes element-wise multiplication: C = A * B
func (ks *KernelSet) DispatchMul(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid size: %d", params.Size)
	}

	return ks.Mul.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, params.C, unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 0, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// DispatchMulScalar executes scalar multiplication: B = A * scalar
func (ks *KernelSet) DispatchMulScalar(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid size: %d", params.Size)
	}

	return ks.MulScalar.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, unsafe.Pointer(&params.Scalar), unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 4, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// DispatchSiLU executes SiLU activation
func (ks *KernelSet) DispatchSiLU(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid size: %d", params.Size)
	}

	return ks.SiLU.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}

// DispatchCopy executes buffer copy
func (ks *KernelSet) DispatchCopy(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	if params.Size == 0 {
		return fmt.Errorf("invalid size: %d", params.Size)
	}

	return ks.Copy.Dispatch(queuePtr, DispatchParams{
		Buffers:     []unsafe.Pointer{params.A, params.B, unsafe.Pointer(&params.Size)},
		BufferSizes: []uint64{0, 0, 4},
		ThreadsX:    uint(params.Size),
		ThreadsY:    1,
		ThreadsZ:    1,
	})
}
