// +build !darwin

package metal

import (
	"fmt"
	"unsafe"
)

// Library stub for non-Darwin platforms
type Library struct{}

func CompileLibrary(devicePtr unsafe.Pointer) (*Library, error) {
	return nil, fmt.Errorf("Metal is only supported on macOS")
}

func (l *Library) Free() {}

func (l *Library) CreatePipeline(functionName string) (*Pipeline, error) {
	return nil, fmt.Errorf("Metal is only supported on macOS")
}

// Pipeline stub
type Pipeline struct{}

func (p *Pipeline) Free() {}

func (p *Pipeline) Dispatch(queuePtr unsafe.Pointer, params DispatchParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

// DispatchParams stub
type DispatchParams struct {
	Buffers     []unsafe.Pointer
	BufferSizes []uint64
	ThreadsX    uint
	ThreadsY    uint
	ThreadsZ    uint
}

// KernelSet stub
type KernelSet struct{}

func NewKernelSet(devicePtr unsafe.Pointer) (*KernelSet, error) {
	return nil, fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) Free() {}

// Stub parameter types
type MatMulParams struct {
	A         unsafe.Pointer
	B         unsafe.Pointer
	C         unsafe.Pointer
	M, N, K   uint32
	SingleRow bool
}

type SoftmaxParams struct {
	Input  unsafe.Pointer
	Output unsafe.Pointer
	Size   uint32
}

type SoftmaxBatchedParams struct {
	Input     unsafe.Pointer
	Output    unsafe.Pointer
	BatchSize uint32
	SeqLen    uint32
}

type RMSNormParams struct {
	Input   unsafe.Pointer
	Weight  unsafe.Pointer
	Output  unsafe.Pointer
	Size    uint32
	Epsilon float32
}

type RMSNormBatchedParams struct {
	Input     unsafe.Pointer
	Weight    unsafe.Pointer
	Output    unsafe.Pointer
	BatchSize uint32
	Dim       uint32
	Epsilon   float32
}

type ElementwiseParams struct {
	A      unsafe.Pointer
	B      unsafe.Pointer
	C      unsafe.Pointer
	Size   uint32
	Scalar float32
}

// Stub methods
func (ks *KernelSet) DispatchMatMul(queuePtr unsafe.Pointer, params MatMulParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchSoftmax(queuePtr unsafe.Pointer, params SoftmaxParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchSoftmaxBatched(queuePtr unsafe.Pointer, params SoftmaxBatchedParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchRMSNorm(queuePtr unsafe.Pointer, params RMSNormParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchRMSNormBatched(queuePtr unsafe.Pointer, params RMSNormBatchedParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchAdd(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchMul(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchMulScalar(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchSiLU(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}

func (ks *KernelSet) DispatchCopy(queuePtr unsafe.Pointer, params ElementwiseParams) error {
	return fmt.Errorf("Metal is only supported on macOS")
}
