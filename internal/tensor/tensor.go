package tensor

import (
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/xupit3r/vibrant/internal/gpu"
	"github.com/xupit3r/vibrant/internal/gpu/metal"
)

// DataType represents the data type of tensor elements
type DataType int

const (
	Float32 DataType = iota // 32-bit floating point
	Float16                 // 16-bit floating point
	Q4_K                    // 4-bit k-quant
	Q5_K                    // 5-bit k-quant
	Q6_K                    // 6-bit k-quant
	Q8_0                    // 8-bit quantization
)

// String returns the name of the data type
func (dt DataType) String() string {
	switch dt {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case Q4_K:
		return "q4_k"
	case Q5_K:
		return "q5_k"
	case Q6_K:
		return "q6_k"
	case Q8_0:
		return "q8_0"
	default:
		return "unknown"
	}
}

// BytesPerElement returns the number of bytes per element for this type
func (dt DataType) BytesPerElement() int {
	switch dt {
	case Float32:
		return 4
	case Float16:
		return 2
	case Q8_0:
		return 1 // Approximate (actual is quantized blocks)
	case Q4_K, Q5_K, Q6_K:
		return 1 // Approximate (actual is quantized blocks)
	default:
		return 4
	}
}

// Device represents the compute device
type Device int

const (
	CPU Device = iota // CPU device
	GPU               // GPU device (future)
)

// String returns the name of the device
func (d Device) String() string {
	switch d {
	case CPU:
		return "cpu"
	case GPU:
		return "gpu"
	default:
		return "unknown"
	}
}

// Tensor represents a multi-dimensional array
type Tensor struct {
	data   interface{} // Underlying data ([]float32, []float16, []uint8, etc.)
	shape  []int       // Dimensions [batch, seq_len, hidden_dim, ...]
	stride []int       // Memory layout strides for indexing
	dtype  DataType    // Data type
	device Device      // Compute device
	offset int         // Offset in data array (for views/slices)

	// Memory-mapped file info (if applicable)
	mmapFile *os.File
	mmapData []byte

	// Optimization flags
	transposed bool // True if this is a pre-transposed weight matrix (for matmul optimization)
	pooled     bool // True if allocated from tensor pool (should be returned via PutTensor)

	// Weight cache (for avoiding redundant dequantization)
	dequantCache *Tensor // Cached Float32 pre-transposed copy
	cacheGen     uint64  // LRU generation counter

	// GPU support
	gpuBuffer  gpu.Buffer      // GPU buffer if tensor is on GPU
	gpuDevice  gpu.Device      // GPU device reference
	gpuKernels *metal.KernelSet // Compiled Metal kernels (shared across tensors on same device)
}

// NewTensor creates a new tensor with the given shape and data type
func NewTensor(shape []int, dtype DataType) *Tensor {
	// Calculate total elements
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	// Allocate data based on type
	var data interface{}
	switch dtype {
	case Float32:
		data = make([]float32, size)
	case Float16:
		data = make([]uint16, size) // float16 stored as uint16
	case Q8_0, Q4_K, Q5_K, Q6_K:
		// For quantized types, allocate enough bytes
		// Actual allocation depends on block structure
		data = make([]uint8, size)
	default:
		data = make([]float32, size)
	}

	return &Tensor{
		data:   data,
		shape:  shape,
		stride: computeStrides(shape),
		dtype:  dtype,
		device: CPU,
		offset: 0,
	}
}

// NewTensorFromData creates a tensor from existing data
func NewTensorFromData(data interface{}, shape []int) *Tensor {
	// Infer data type from data
	var dtype DataType
	switch data.(type) {
	case []float32:
		dtype = Float32
	case []uint16:
		dtype = Float16
	case []uint8:
		dtype = Q8_0 // Default for byte arrays
	default:
		dtype = Float32
	}

	return &Tensor{
		data:   data,
		shape:  shape,
		stride: computeStrides(shape),
		dtype:  dtype,
		device: CPU,
		offset: 0,
	}
}

// NewTensorMmap creates a memory-mapped tensor from a file
func NewTensorMmap(path string, offset int64, size int64, shape []int, dtype DataType) (*Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	// mmap requires offset to be page-aligned
	pageSize := int64(os.Getpagesize())

	// Calculate page-aligned offset (round down)
	alignedOffset := (offset / pageSize) * pageSize

	// Calculate the page offset (offset within the page)
	pageOffset := int(offset - alignedOffset)

	// Adjust size to include the page offset
	alignedSize := int(size) + pageOffset

	// Memory-map the file region with aligned offset
	mmapData, err := syscall.Mmap(
		int(f.Fd()),
		alignedOffset,
		alignedSize,
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to mmap: %w", err)
	}

	// Adjust mmapData to start at the actual tensor data (skip page offset)
	actualData := mmapData[pageOffset : pageOffset+int(size)]

	// Wrap data based on dtype
	var data interface{}
	switch dtype {
	case Float32:
		// Cast []byte to []float32 using unsafe
		data = (*[1 << 30]float32)(unsafe.Pointer(&actualData[0]))[:len(actualData)/4:len(actualData)/4]
	case Float16:
		data = (*[1 << 30]uint16)(unsafe.Pointer(&actualData[0]))[:len(actualData)/2:len(actualData)/2]
	case Q4_K, Q5_K, Q6_K, Q8_0:
		data = actualData
	default:
		data = (*[1 << 30]float32)(unsafe.Pointer(&actualData[0]))[:len(actualData)/4:len(actualData)/4]
	}

	return &Tensor{
		data:     data,
		shape:    shape,
		stride:   computeStrides(shape),
		dtype:    dtype,
		device:   CPU,
		offset:   0,
		mmapFile: f,
		mmapData: mmapData, // Keep the full mmap data for unmapping
	}, nil
}

// Zeros creates a zero-initialized tensor
func Zeros(shape []int, dtype DataType) *Tensor {
	// NewTensor already initializes to zero
	return NewTensor(shape, dtype)
}

// Ones creates a one-initialized tensor
func Ones(shape []int, dtype DataType) *Tensor {
	t := NewTensor(shape, dtype)

	// Set all elements to 1
	switch dtype {
	case Float32:
		data := t.data.([]float32)
		for i := range data {
			data[i] = 1.0
		}
	case Float16:
		data := t.data.([]uint16)
		one := float32ToFloat16(1.0)
		for i := range data {
			data[i] = one
		}
	}

	return t
}

// computeStrides calculates memory layout strides for the given shape
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	if len(shape) == 0 {
		return strides
	}

	// Row-major order (C-style)
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return strides
}

// Shape returns the tensor's shape
func (t *Tensor) Shape() []int {
	return t.shape
}

// Stride returns the tensor's strides
func (t *Tensor) Stride() []int {
	return t.stride
}

// DType returns the tensor's data type
func (t *Tensor) DType() DataType {
	return t.dtype
}

// Device returns the tensor's device
func (t *Tensor) Device() Device {
	return t.device
}

// Size returns the total number of elements
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.shape {
		size *= dim
	}
	return size
}

// NumDims returns the number of dimensions
func (t *Tensor) NumDims() int {
	return len(t.shape)
}

// At returns the value at the given multi-dimensional index
func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("At() expected %d indices (shape=%v), got %d indices=%v", len(t.shape), t.shape, len(indices), indices))
	}

	// Compute linear index
	idx := t.offset
	for i, index := range indices {
		if index < 0 || index >= t.shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for dimension %d (size %d)", index, i, t.shape[i]))
		}
		idx += index * t.stride[i]
	}

	// Return value based on dtype
	switch t.dtype {
	case Float32:
		return t.data.([]float32)[idx]
	case Float16:
		return float16ToFloat32(t.data.([]uint16)[idx])
	case Q4_K:
		return DequantizeQ4_KElement(t.data.([]byte), idx)
	case Q5_K:
		return DequantizeQ5_KElement(t.data.([]byte), idx)
	case Q6_K:
		return DequantizeQ6_KElement(t.data.([]byte), idx)
	default:
		panic(fmt.Sprintf("At() not supported for dtype %s", t.dtype))
	}
}

// Set sets the value at the given multi-dimensional index
func (t *Tensor) Set(value float32, indices ...int) {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.shape), len(indices)))
	}

	// Compute linear index
	idx := t.offset
	for i, index := range indices {
		if index < 0 || index >= t.shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for dimension %d (size %d)", index, i, t.shape[i]))
		}
		idx += index * t.stride[i]
	}

	// Set value based on dtype
	switch t.dtype {
	case Float32:
		t.data.([]float32)[idx] = value
	case Float16:
		t.data.([]uint16)[idx] = float32ToFloat16(value)
	default:
		panic(fmt.Sprintf("Set() not supported for dtype %s", t.dtype))
	}
}

// Data returns the underlying data (use with caution)
func (t *Tensor) Data() interface{} {
	return t.data
}

// Float32Data returns the underlying float32 slice directly.
// Panics if the tensor's data is not []float32.
func (t *Tensor) Float32Data() []float32 {
	return t.data.([]float32)
}

// GetOrDequantTranspose returns a cached dequantized and pre-transposed copy of this tensor.
// For quantized weight tensors, this avoids redundant dequantization and transposition.
// For Float32 tensors, if already pre-transposed, returns the tensor itself unchanged.
func (t *Tensor) GetOrDequantTranspose() *Tensor {
	// If already Float32 and pre-transposed, return as-is (no work needed)
	if t.dtype == Float32 && t.transposed {
		return t
	}

	// If Float32 but not transposed, return as-is for backward compatibility
	// (caller may handle transpose or this is activation data, not weights)
	if t.dtype == Float32 {
		return t
	}

	// Return cached version if available (cache hit)
	if t.dequantCache != nil {
		t.cacheGen = nextCacheGen()
		atomic.AddUint64(&DefaultWeightCache.hits, 1)
		return t.dequantCache
	}

	// Cache miss - need to dequantize and transpose
	atomic.AddUint64(&DefaultWeightCache.misses, 1)

	// Dequantize to Float32
	dequant := dequantIfNeeded(t)

	// Pre-transpose for matmul (B matrix is accessed column-wise)
	transposed := dequant.Transpose()
	transposed.MarkTransposed()

	// Register with cache manager (may evict others)
	DefaultWeightCache.Register(t, transposed)
	t.dequantCache = transposed
	t.cacheGen = nextCacheGen()

	return transposed
}

// Close releases resources (unmaps memory if needed)
func (t *Tensor) Close() error {
	if t.mmapFile != nil {
		if err := syscall.Munmap(t.mmapData); err != nil {
			return err
		}
		if err := t.mmapFile.Close(); err != nil {
			return err
		}
		t.mmapFile = nil
		t.mmapData = nil
	}
	return nil
}

// float16 conversion helpers (simplified IEEE 754 half-precision)
func float32ToFloat16(f float32) uint16 {
	// Handle special case of 0.0
	if f == 0.0 {
		return 0
	}

	// Simplified conversion (not full IEEE 754)
	// For now, just truncate mantissa
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := (bits >> 31) & 0x1
	exp := int32((bits>>23)&0xFF) - 127 + 15
	mantissa := (bits >> 13) & 0x3FF

	if exp <= 0 {
		return uint16(sign << 15) // Underflow to zero
	}
	if exp >= 31 {
		return uint16((sign << 15) | 0x7C00) // Overflow to infinity
	}

	return uint16((sign << 15) | (uint32(exp) << 10) | mantissa)
}

func float16ToFloat32(h uint16) float32 {
	// Simplified conversion (not full IEEE 754)
	sign := uint32((h >> 15) & 0x1)
	exp := uint32((h >> 10) & 0x1F)
	mantissa := uint32(h & 0x3FF)

	if exp == 0 {
		// Zero or subnormal
		return 0.0
	}
	if exp == 31 {
		// Infinity or NaN
		if mantissa == 0 {
			if sign == 1 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}

	// Normal number
	exp = exp - 15 + 127
	bits := (sign << 31) | (exp << 23) | (mantissa << 13)
	return *(*float32)(unsafe.Pointer(&bits))
}

// IsTransposed returns true if this tensor is marked as pre-transposed
func (t *Tensor) IsTransposed() bool {
return t.transposed
}

// MarkTransposed marks this tensor as pre-transposed for matmul optimization
func (t *Tensor) MarkTransposed() {
t.transposed = true
}

// Transpose creates a transposed copy of a 2D tensor (optimized for cache locality)
// For a matrix of shape [M, N], returns shape [N, M] with transposed data
func (t *Tensor) Transpose() *Tensor {
if len(t.shape) != 2 {
panic("Transpose only supports 2D tensors")
}

M, N := t.shape[0], t.shape[1]

// Create transposed tensor with swapped dimensions
result := NewTensor([]int{N, M}, t.dtype)
result.transposed = true // Mark as transposed

switch t.dtype {
case Float32:
src := t.data.([]float32)
dst := result.data.([]float32)

// Cache-blocked transpose for better performance
// Process 32×32 tiles to fit in L1 cache
const blockSize = 32

for i := 0; i < M; i += blockSize {
for j := 0; j < N; j += blockSize {
// Transpose this block
iEnd := min(i+blockSize, M)
jEnd := min(j+blockSize, N)

for ii := i; ii < iEnd; ii++ {
for jj := j; jj < jEnd; jj++ {
dst[jj*M+ii] = src[ii*N+jj]
}
}
}
}

case Q4_K, Q5_K, Q6_K, Q8_0:
// For quantized tensors, we can't transpose the data directly
// Instead, mark as transposed and the matmul will handle it
// This is more complex - for now, panic
panic("Transpose not yet implemented for quantized tensors")

default:
panic(fmt.Sprintf("Transpose not implemented for dtype %v", t.dtype))
}

return result
}

// PretransposeInPlace transposes this tensor in-place and marks it as pre-transposed.
// This is an optimization for weight matrices that are used repeatedly in matmul operations.
// Only works for Float32 2D tensors. Returns error if tensor is not suitable for in-place transpose.
func (t *Tensor) PretransposeInPlace() error {
	if len(t.shape) != 2 {
		return fmt.Errorf("PretransposeInPlace only supports 2D tensors, got %dD", len(t.shape))
	}

	if t.dtype != Float32 {
		return fmt.Errorf("PretransposeInPlace only supports Float32 tensors, got %s", t.dtype)
	}

	if t.transposed {
		// Already transposed, nothing to do
		return nil
	}

	// Create a transposed copy
	transposed := t.Transpose()

	// Replace this tensor's data with the transposed data
	t.data = transposed.data
	t.shape = transposed.shape
	t.stride = transposed.stride
	t.transposed = true

	return nil
}

