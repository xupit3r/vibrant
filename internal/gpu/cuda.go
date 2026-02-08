// +build linux,cgo

package gpu

/*
#cgo CFLAGS: -I/opt/cuda/include -I/usr/local/cuda/include
#cgo LDFLAGS: -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart

#include <cuda_runtime.h>
#include <stdlib.h>

// Error checking helper
static const char* getCudaErrorString(cudaError_t error) {
    return cudaGetErrorString(error);
}
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// CUDADevice represents a CUDA GPU device
type CUDADevice struct {
	deviceID int
	name     string
	buffers  map[uintptr]*cudaBuffer
	mu       sync.RWMutex
	pool     *BufferPool // Buffer pool for efficient allocation
}

// Singleton CUDA device — creating multiple CUDA contexts wastes GPU memory
var (
	cudaDeviceSingleton *CUDADevice
	cudaDeviceOnce      sync.Once
	cudaDeviceErr       error
)

// NewCUDADevice returns the singleton CUDA device (created on first call)
func NewCUDADevice() (*CUDADevice, error) {
	cudaDeviceOnce.Do(func() {
		cudaDeviceSingleton, cudaDeviceErr = initCUDADevice()
	})
	return cudaDeviceSingleton, cudaDeviceErr
}

// initCUDADevice creates and initializes the CUDA device
func initCUDADevice() (*CUDADevice, error) {
	// Check if CUDA is available
	var deviceCount C.int
	err := C.cudaGetDeviceCount(&deviceCount)
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("CUDA not available: %s", C.GoString(C.getCudaErrorString(err)))
	}

	if deviceCount == 0 {
		return nil, fmt.Errorf("no CUDA devices found")
	}

	// Use device 0 by default (TODO: support device selection in Phase 11.4)
	deviceID := 0
	err = C.cudaSetDevice(C.int(deviceID))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("failed to set CUDA device %d: %s", deviceID, C.GoString(C.getCudaErrorString(err)))
	}

	// Get device properties
	var props C.struct_cudaDeviceProp
	err = C.cudaGetDeviceProperties(&props, C.int(deviceID))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("failed to get device properties: %s", C.GoString(C.getCudaErrorString(err)))
	}

	name := C.GoString(&props.name[0])

	dev := &CUDADevice{
		deviceID: deviceID,
		name:     name,
		buffers:  make(map[uintptr]*cudaBuffer),
	}

	// Initialize buffer pool for efficient allocation reuse
	var free, total C.size_t
	err = C.cudaMemGetInfo(&free, &total)
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("failed to get memory info: %s", C.GoString(C.getCudaErrorString(err)))
	}

	// Use 90% of free memory for pool
	poolMaxBytes := int64(free) * 9 / 10
	dev.pool = NewBufferPool(dev, poolMaxBytes)

	return dev, nil
}

func (d *CUDADevice) Type() DeviceType {
	return DeviceTypeGPU
}

func (d *CUDADevice) Name() string {
	return d.name
}

func (d *CUDADevice) Allocate(size int64) (Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("invalid buffer size: %d", size)
	}

	// Use pool if available
	if d.pool != nil {
		return d.pool.Allocate(size)
	}

	// Direct allocation (fallback if no pool)
	return d.allocateDirect(size)
}

// allocateDirect performs direct buffer allocation without pooling
func (d *CUDADevice) allocateDirect(size int64) (Buffer, error) {
	var ptr unsafe.Pointer
	err := C.cudaMalloc(&ptr, C.size_t(size))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("failed to allocate CUDA buffer of size %d: %s", size, C.GoString(C.getCudaErrorString(err)))
	}

	buf := &cudaBuffer{
		ptr:    ptr,
		size:   size,
		device: d,
	}

	d.mu.Lock()
	d.buffers[uintptr(ptr)] = buf
	d.mu.Unlock()

	return buf, nil
}

func (d *CUDADevice) Copy(dst, src Buffer, size int64) error {
	// Unwrap pooled buffers if necessary
	dstUnwrapped := dst
	if pooled, ok := dst.(*pooledBuffer); ok {
		dstUnwrapped = pooled.Buffer
	}
	srcUnwrapped := src
	if pooled, ok := src.(*pooledBuffer); ok {
		srcUnwrapped = pooled.Buffer
	}

	dstBuf, ok := dstUnwrapped.(*cudaBuffer)
	if !ok {
		return fmt.Errorf("dst is not a CUDA buffer")
	}
	srcBuf, ok := srcUnwrapped.(*cudaBuffer)
	if !ok {
		return fmt.Errorf("src is not a CUDA buffer")
	}

	if size > dstBuf.size || size > srcBuf.size {
		return fmt.Errorf("copy size %d exceeds buffer size (dst: %d, src: %d)",
			size, dstBuf.size, srcBuf.size)
	}

	// Device-to-device copy
	err := C.cudaMemcpy(dstBuf.ptr, srcBuf.ptr, C.size_t(size), C.cudaMemcpyDeviceToDevice)
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to copy CUDA buffer: %s", C.GoString(C.getCudaErrorString(err)))
	}

	return nil
}

func (d *CUDADevice) Sync() error {
	err := C.cudaDeviceSynchronize()
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to synchronize CUDA device: %s", C.GoString(C.getCudaErrorString(err)))
	}
	return nil
}

func (d *CUDADevice) Free() error {
	// Clear buffer pool first (before acquiring lock to avoid deadlock)
	if d.pool != nil {
		d.pool.Clear()
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Free all remaining buffers
	for _, buf := range d.buffers {
		if buf.ptr != nil {
			C.cudaFree(buf.ptr)
			buf.ptr = nil
		}
	}
	d.buffers = nil

	// Reset device
	err := C.cudaDeviceReset()
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to reset CUDA device: %s", C.GoString(C.getCudaErrorString(err)))
	}

	return nil
}

func (d *CUDADevice) MemoryUsage() (int64, int64) {
	var free, total C.size_t
	err := C.cudaMemGetInfo(&free, &total)
	if err != C.cudaSuccess {
		return 0, 0
	}

	used := int64(total) - int64(free)
	return used, int64(total)
}

// PoolStats returns buffer pool statistics
func (d *CUDADevice) PoolStats() PoolStats {
	if d.pool != nil {
		return d.pool.Stats()
	}
	return PoolStats{}
}

// PoolMemoryUsage returns buffer pool memory usage
func (d *CUDADevice) PoolMemoryUsage() (pooled, active, max int64) {
	if d.pool != nil {
		return d.pool.MemoryUsage()
	}
	return 0, 0, 0
}

// cudaBuffer implements Buffer for CUDA GPU memory
type cudaBuffer struct {
	ptr    unsafe.Pointer
	size   int64
	device *CUDADevice
	mu     sync.RWMutex
}

func (b *cudaBuffer) Size() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}

func (b *cudaBuffer) Ptr() uintptr {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return uintptr(b.ptr)
}

func (b *cudaBuffer) CopyToHost(dst []byte) error {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if int64(len(dst)) < b.size {
		return fmt.Errorf("destination buffer too small: %d < %d", len(dst), b.size)
	}

	// Device to host copy
	err := C.cudaMemcpy(unsafe.Pointer(&dst[0]), b.ptr, C.size_t(b.size), C.cudaMemcpyDeviceToHost)
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to copy to host: %s", C.GoString(C.getCudaErrorString(err)))
	}

	return nil
}

func (b *cudaBuffer) CopyFromHost(src []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.size < int64(len(src)) {
		return fmt.Errorf("buffer too small: %d < %d", b.size, len(src))
	}

	// Host to device copy
	err := C.cudaMemcpy(b.ptr, unsafe.Pointer(&src[0]), C.size_t(len(src)), C.cudaMemcpyHostToDevice)
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to copy from host: %s", C.GoString(C.getCudaErrorString(err)))
	}

	return nil
}

func (b *cudaBuffer) Free() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.ptr != nil {
		// Remove from device tracking
		if b.device != nil {
			b.device.mu.Lock()
			delete(b.device.buffers, uintptr(b.ptr))
			b.device.mu.Unlock()
		}

		err := C.cudaFree(b.ptr)
		if err != C.cudaSuccess {
			return fmt.Errorf("failed to free CUDA buffer: %s", C.GoString(C.getCudaErrorString(err)))
		}
		b.ptr = nil
	}

	return nil
}

func (b *cudaBuffer) Device() Device {
	return b.device
}

func (b *cudaBuffer) MetalBuffer() unsafe.Pointer {
	return nil // Not a Metal buffer
}
