// +build darwin,cgo

package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>

// Compile Metal library from source code
void* compileLibrary(void* device, const char* source, char** error) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        NSString* sourceString = [NSString stringWithUTF8String:source];
        
        NSError* compileError = nil;
        id<MTLLibrary> library = [mtlDevice newLibraryWithSource:sourceString
                                                         options:nil
                                                           error:&compileError];
        
        if (library == nil) {
            if (error != NULL && compileError != nil) {
                NSString* errorStr = [compileError localizedDescription];
                *error = strdup([errorStr UTF8String]);
            }
            return NULL;
        }
        
        return (void*)CFBridgingRetain(library);
    }
}

// Get function from library
void* getFunction(void* library, const char* name, char** error) {
    @autoreleasepool {
        id<MTLLibrary> mtlLibrary = (__bridge id<MTLLibrary>)library;
        NSString* nameString = [NSString stringWithUTF8String:name];
        
        id<MTLFunction> function = [mtlLibrary newFunctionWithName:nameString];
        if (function == nil) {
            if (error != NULL) {
                *error = strdup("Function not found in library");
            }
            return NULL;
        }
        
        return (void*)CFBridgingRetain(function);
    }
}

// Create compute pipeline state
void* createPipeline(void* device, void* function, char** error) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLFunction> mtlFunction = (__bridge id<MTLFunction>)function;
        
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipeline = 
            [mtlDevice newComputePipelineStateWithFunction:mtlFunction
                                                      error:&pipelineError];
        
        if (pipeline == nil) {
            if (error != NULL && pipelineError != nil) {
                NSString* errorStr = [pipelineError localizedDescription];
                *error = strdup([errorStr UTF8String]);
            }
            return NULL;
        }
        
        return (void*)CFBridgingRetain(pipeline);
    }
}

// Dispatch compute kernel
void dispatchKernel(
    void* commandQueue,
    void* pipeline,
    void* buffers[],
    size_t* bufferSizes,
    uint numBuffers,
    uint threadsX,
    uint threadsY,
    uint threadsZ)
{
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLComputePipelineState> pipelineState = (__bridge id<MTLComputePipelineState>)pipeline;
        
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        
        // Set buffers
        for (uint i = 0; i < numBuffers; i++) {
            if (bufferSizes[i] == 0) {
                // Size 0 means it's a Metal buffer object
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
                [encoder setBuffer:buffer offset:0 atIndex:i];
            } else if (bufferSizes[i] <= 32) {
                // Small data: use setBytes for efficiency
                [encoder setBytes:buffers[i] length:bufferSizes[i] atIndex:i];
            } else {
                // This shouldn't happen - large data should be buffers with size 0
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
                [encoder setBuffer:buffer offset:0 atIndex:i];
            }
        }
        
        // Calculate thread groups
        NSUInteger maxThreads = [pipelineState maxTotalThreadsPerThreadgroup];
        NSUInteger threadGroupSize = MIN(256, maxThreads);
        
        MTLSize gridSize = MTLSizeMake(threadsX, threadsY, threadsZ);
        MTLSize threadgroupSize;
        
        if (threadsY > 1) {
            // 2D dispatch (e.g., matrix operations)
            uint tgX = MIN(16, threadsX);
            uint tgY = MIN(16, threadsY);
            threadgroupSize = MTLSizeMake(tgX, tgY, 1);
        } else {
            // 1D dispatch
            threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        }
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// Free library
void freeLibrary(void* library) {
    if (library != NULL) {
        CFBridgingRelease(library);
    }
}

// Free function
void freeFunction(void* function) {
    if (function != NULL) {
        CFBridgingRelease(function);
    }
}

// Free pipeline
void freePipeline(void* pipeline) {
    if (pipeline != NULL) {
        CFBridgingRelease(pipeline);
    }
}
*/
import "C"
import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:embed kernels.metal
var metalSource string

// Library represents a compiled Metal library
type Library struct {
	ptr    unsafe.Pointer
	device unsafe.Pointer
}

// CompileLibrary compiles the embedded Metal shader code
func CompileLibrary(devicePtr unsafe.Pointer) (*Library, error) {
	if devicePtr == nil {
		return nil, fmt.Errorf("device pointer is nil")
	}

	sourceC := C.CString(metalSource)
	defer C.free(unsafe.Pointer(sourceC))

	var errorC *C.char
	libPtr := C.compileLibrary(devicePtr, sourceC, &errorC)
	
	if libPtr == nil {
		errorMsg := "failed to compile Metal library"
		if errorC != nil {
			errorMsg = C.GoString(errorC)
			C.free(unsafe.Pointer(errorC))
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &Library{
		ptr:    libPtr,
		device: devicePtr,
	}, nil
}

// Free releases the library
func (l *Library) Free() {
	if l.ptr != nil {
		C.freeLibrary(l.ptr)
		l.ptr = nil
	}
}

// Pipeline represents a compiled compute pipeline
type Pipeline struct {
	ptr  unsafe.Pointer
	name string
}

// CreatePipeline creates a compute pipeline for the given kernel function
func (l *Library) CreatePipeline(functionName string) (*Pipeline, error) {
	nameC := C.CString(functionName)
	defer C.free(unsafe.Pointer(nameC))

	var errorC *C.char
	funcPtr := C.getFunction(l.ptr, nameC, &errorC)
	
	if funcPtr == nil {
		errorMsg := fmt.Sprintf("function '%s' not found", functionName)
		if errorC != nil {
			errorMsg = C.GoString(errorC)
			C.free(unsafe.Pointer(errorC))
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}
	defer C.freeFunction(funcPtr)

	pipelinePtr := C.createPipeline(l.device, funcPtr, &errorC)
	
	if pipelinePtr == nil {
		errorMsg := fmt.Sprintf("failed to create pipeline for '%s'", functionName)
		if errorC != nil {
			errorMsg = C.GoString(errorC)
			C.free(unsafe.Pointer(errorC))
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &Pipeline{
		ptr:  pipelinePtr,
		name: functionName,
	}, nil
}

// Free releases the pipeline
func (p *Pipeline) Free() {
	if p.ptr != nil {
		C.freePipeline(p.ptr)
		p.ptr = nil
	}
}

// DispatchParams contains parameters for kernel dispatch
type DispatchParams struct {
	Buffers     []unsafe.Pointer // Buffer pointers (MTLBuffer or raw data for small buffers)
	BufferSizes []uint64         // Size of each buffer (for determining setBytes vs setBuffer)
	ThreadsX    uint             // Number of threads in X dimension
	ThreadsY    uint             // Number of threads in Y dimension (default 1)
	ThreadsZ    uint             // Number of threads in Z dimension (default 1)
}

// Dispatch executes the compute kernel with the given parameters
func (p *Pipeline) Dispatch(queuePtr unsafe.Pointer, params DispatchParams) error {
	if p.ptr == nil {
		return fmt.Errorf("pipeline is nil")
	}
	if queuePtr == nil {
		return fmt.Errorf("command queue is nil")
	}

	numBuffers := len(params.Buffers)
	if numBuffers != len(params.BufferSizes) {
		return fmt.Errorf("buffer count mismatch: %d buffers, %d sizes", 
			numBuffers, len(params.BufferSizes))
	}

	if params.ThreadsX == 0 {
		return fmt.Errorf("ThreadsX must be > 0")
	}
	if params.ThreadsY == 0 {
		params.ThreadsY = 1
	}
	if params.ThreadsZ == 0 {
		params.ThreadsZ = 1
	}

	if numBuffers == 0 {
		// No buffers case
		C.dispatchKernel(
			queuePtr,
			p.ptr,
			nil,
			nil,
			0,
			C.uint(params.ThreadsX),
			C.uint(params.ThreadsY),
			C.uint(params.ThreadsZ),
		)
		return nil
	}

	// Allocate C arrays (not pinned by Go GC)
	buffersC := C.malloc(C.size_t(numBuffers) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(buffersC)
	
	sizesC := C.malloc(C.size_t(numBuffers) * C.size_t(unsafe.Sizeof(C.size_t(0))))
	defer C.free(sizesC)

	// Fill the arrays
	bufferArray := (*[1 << 28]unsafe.Pointer)(buffersC)[:numBuffers:numBuffers]
	sizeArray := (*[1 << 28]C.size_t)(sizesC)[:numBuffers:numBuffers]
	
	for i := range params.Buffers {
		bufferArray[i] = params.Buffers[i]
		sizeArray[i] = C.size_t(params.BufferSizes[i])
	}

	C.dispatchKernel(
		queuePtr,
		p.ptr,
		(*unsafe.Pointer)(buffersC),
		(*C.size_t)(sizesC),
		C.uint(numBuffers),
		C.uint(params.ThreadsX),
		C.uint(params.ThreadsY),
		C.uint(params.ThreadsZ),
	)

	return nil
}
