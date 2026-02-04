// +build darwin

package gpu

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdlib.h>

// Helper functions to avoid direct Objective-C in Go

typedef struct {
    void* device;
    void* commandQueue;
} MetalContext;

// Create Metal device and command queue
MetalContext* createMetalContext() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return NULL;
        }
        
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return NULL;
        }
        
        MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
        ctx->device = (void*)CFBridgingRetain(device);
        ctx->commandQueue = (void*)CFBridgingRetain(queue);
        return ctx;
    }
}

// Get device name (caller must free returned string)
char* getDeviceName(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        NSString* name = [mtlDevice name];
        const char* utf8 = [name UTF8String];
        return strdup(utf8);
    }
}

// Allocate buffer
void* allocateBuffer(void* device, size_t size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:size 
                                                      options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return NULL;
        }
        return (void*)CFBridgingRetain(buffer);
    }
}

// Free buffer
void freeBuffer(void* buffer) {
    if (buffer != NULL) {
        CFBridgingRelease(buffer);
    }
}

// Get buffer contents pointer
void* getBufferContents(void* buffer) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        return [mtlBuffer contents];
    }
}

// Get buffer length
size_t getBufferLength(void* buffer) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        return [mtlBuffer length];
    }
}

// Synchronize (wait for all GPU operations to complete)
void synchronize(void* commandQueue) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// Get recommended max working set size
size_t getRecommendedMaxWorkingSetSize(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return [mtlDevice recommendedMaxWorkingSetSize];
    }
}

// Get current allocated size
size_t getCurrentAllocatedSize(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return [mtlDevice currentAllocatedSize];
    }
}

// Free Metal context
void freeMetalContext(MetalContext* ctx) {
    if (ctx != NULL) {
        if (ctx->commandQueue != NULL) {
            CFBridgingRelease(ctx->commandQueue);
        }
        if (ctx->device != NULL) {
            CFBridgingRelease(ctx->device);
        }
        free(ctx);
    }
}
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// MetalDevice represents a Metal GPU device
type MetalDevice struct {
	ctx     *C.MetalContext
	name    string
	buffers map[uintptr]*metalBuffer
	mu      sync.RWMutex
}

// NewMetalDevice creates a new Metal device
func NewMetalDevice() (*MetalDevice, error) {
	ctx := C.createMetalContext()
	if ctx == nil {
		return nil, fmt.Errorf("failed to create Metal device (Metal not available)")
	}

	namePtr := C.getDeviceName(ctx.device)
	name := C.GoString(namePtr)
	C.free(unsafe.Pointer(namePtr))

	return &MetalDevice{
		ctx:     ctx,
		name:    name,
		buffers: make(map[uintptr]*metalBuffer),
	}, nil
}

func (d *MetalDevice) Type() DeviceType {
	return DeviceTypeGPU
}

func (d *MetalDevice) Name() string {
	return d.name
}

func (d *MetalDevice) Allocate(size int64) (Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("invalid buffer size: %d", size)
	}

	bufPtr := C.allocateBuffer(d.ctx.device, C.size_t(size))
	if bufPtr == nil {
		return nil, fmt.Errorf("failed to allocate Metal buffer of size %d", size)
	}

	buf := &metalBuffer{
		ptr:    bufPtr,
		size:   size,
		device: d,
	}

	d.mu.Lock()
	d.buffers[uintptr(bufPtr)] = buf
	d.mu.Unlock()

	return buf, nil
}

func (d *MetalDevice) Copy(dst, src Buffer, size int64) error {
	dstBuf, ok := dst.(*metalBuffer)
	if !ok {
		return fmt.Errorf("dst is not a Metal buffer")
	}
	srcBuf, ok := src.(*metalBuffer)
	if !ok {
		return fmt.Errorf("src is not a Metal buffer")
	}

	if size > dstBuf.size || size > srcBuf.size {
		return fmt.Errorf("copy size %d exceeds buffer size (dst: %d, src: %d)", 
			size, dstBuf.size, srcBuf.size)
	}

	// Use Metal's unified memory - just memcpy the contents
	dstContents := C.getBufferContents(dstBuf.ptr)
	srcContents := C.getBufferContents(srcBuf.ptr)
	
	if dstContents == nil || srcContents == nil {
		return fmt.Errorf("failed to get buffer contents")
	}

	C.memcpy(dstContents, srcContents, C.size_t(size))

	return nil
}

func (d *MetalDevice) Sync() error {
	if d.ctx == nil || d.ctx.commandQueue == nil {
		return fmt.Errorf("invalid Metal context")
	}
	C.synchronize(d.ctx.commandQueue)
	return nil
}

func (d *MetalDevice) Free() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Free all buffers
	for _, buf := range d.buffers {
		if buf.ptr != nil {
			C.freeBuffer(buf.ptr)
			buf.ptr = nil
		}
	}
	d.buffers = nil

	// Free context
	if d.ctx != nil {
		C.freeMetalContext(d.ctx)
		d.ctx = nil
	}

	return nil
}

func (d *MetalDevice) MemoryUsage() (int64, int64) {
	if d.ctx == nil || d.ctx.device == nil {
		return 0, 0
	}

	used := int64(C.getCurrentAllocatedSize(d.ctx.device))
	total := int64(C.getRecommendedMaxWorkingSetSize(d.ctx.device))

	return used, total
}

// metalBuffer implements Buffer for Metal GPU memory
type metalBuffer struct {
	ptr    unsafe.Pointer
	size   int64
	device *MetalDevice
	mu     sync.RWMutex
}

func (b *metalBuffer) Size() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}

func (b *metalBuffer) Ptr() uintptr {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return uintptr(b.ptr)
}

func (b *metalBuffer) CopyToHost(dst []byte) error {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if int64(len(dst)) < b.size {
		return fmt.Errorf("destination buffer too small: %d < %d", len(dst), b.size)
	}

	contents := C.getBufferContents(b.ptr)
	if contents == nil {
		return fmt.Errorf("failed to get buffer contents")
	}

	// Copy from Metal buffer to host
	C.memcpy(unsafe.Pointer(&dst[0]), contents, C.size_t(b.size))

	return nil
}

func (b *metalBuffer) CopyFromHost(src []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.size < int64(len(src)) {
		return fmt.Errorf("buffer too small: %d < %d", b.size, len(src))
	}

	contents := C.getBufferContents(b.ptr)
	if contents == nil {
		return fmt.Errorf("failed to get buffer contents")
	}

	// Copy from host to Metal buffer
	C.memcpy(contents, unsafe.Pointer(&src[0]), C.size_t(len(src)))

	return nil
}

func (b *metalBuffer) Free() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.ptr != nil {
		// Remove from device tracking
		if b.device != nil {
			b.device.mu.Lock()
			delete(b.device.buffers, uintptr(b.ptr))
			b.device.mu.Unlock()
		}

		C.freeBuffer(b.ptr)
		b.ptr = nil
	}

	return nil
}

func (b *metalBuffer) Device() Device {
	return b.device
}
