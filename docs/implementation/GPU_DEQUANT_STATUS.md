# GPU Dequantization - Current Status

**Date**: 2026-02-05  
**Status**: ⚠️ Partially Working - Model loads to GPU but inference hangs

## What Works ✅

1. **Device mapping fixed** - `--device cuda` and `--device metal` now properly recognized
2. **Dequantization implemented** - Q4_K, Q5_K, Q6_K → Float32 conversion working
3. **GPU transfer fixed** - Fixed `unsafe.Slice` issue for tensors > 1GB
4. **Model loads to GPU** - Confirmed 12.7GB VRAM usage (3B model dequantized)
5. **CUDA device detection** - RTX 4090 properly detected and initialized

## What Doesn't Work ❌

**Inference hangs** after "Generating response..." message.

### Timing Analysis

```
[12:07:01] Initializing...
[12:07:07] Moving model weights to GPU...
[12:07:07-14] Dequantizing... (7 seconds for 260+ tensors)
[12:07:14] Model loaded on GPU
[12:07:14] Generating response...
[HANGS INDEFINITELY]
```

### Evidence

1. **GPU Memory**: 12.7GB allocated (up from 0.6GB baseline) ✅
2. **GPU Utilization**: 0-6% during hang (should be 80-100%) ❌
3. **Process State**: Blocked, not using CPU or GPU ❌

## Root Cause Analysis

The first forward pass (inference) is hanging when calling CUDA kernels. Possible causes:

###1. **CUDA Kernel Deadlock**
Most likely. The matmul or other GPU operations are waiting for something that never completes.

### 2. **Stream Synchronization Issue**
CUDA streams may not be properly synchronized between kernel launches.

### 3. **Buffer/Pointer Issue**
GPU buffers may have invalid pointers or sizes causing kernel hangs.

### 4. **Kernel Launch Configuration**
Grid/block dimensions may be incorrect causing infinite loops in kernels.

## Files Modified

1. `internal/llm/engine_custom.go` - Fixed device mapping
2. `internal/tensor/tensor_cuda_api.go` - Added dequantization
3. `internal/tensor/tensor_gpu.go` - Added dequantization (Metal)
4. `internal/tensor/tensor_cuda.go` - Fixed unsafe.Slice for large tensors
5. `internal/tensor/tensor_gpu.go` - Fixed unsafe.Slice for Metal

## Next Steps to Debug

### Immediate Debugging

1. **Add kernel-level logging** in `internal/tensor/ops_cuda.go`:
   ```go
   fmt.Printf("[CUDA] Launching matmul: M=%d, K=%d, N=%d\n", M, K, N)
   err := kernels.LaunchMatMul(...)
   fmt.Printf("[CUDA] Matmul returned: %v\n", err)
   ```

2. **Check kernel launch errors**:
   ```go
   if err := cudaDev.Sync(); err != nil {
       log.Fatalf("CUDA sync failed: %v", err)
   }
   ```

3. **Test individual kernels** with simple test cases:
   ```bash
   go test ./internal/tensor -run TestCUDAMatMul -v
   ```

### Root Cause Investigation

1. **Check if it's specific to matmul** - Test with simpler operations (add, mul)
2. **Verify kernel compilation** - Check `build/cuda/libvibrant_cuda.so` actually contains kernels
3. **Test with smaller matrices** - May be a size-related issue
4. **Check stream handling** - Verify CUDA streams are created/managed correctly

### Alternative Approach

If debugging proves too complex, consider:

1. **Fall back to CPU for inference** - Keep model on GPU but run ops on CPU
2. **Hybrid approach** - Only GPU-accelerate specific large operations
3. **Use llama.cpp CUDA backend** instead of custom kernels

## Workaround for Users

Until fixed, users should use CPU inference:

```bash
# Works perfectly (CPU inference with quantized models)
./vibrant --device cpu chat

# Doesn't work yet (hangs during inference)
./vibrant --device cuda chat
```

## Testing Commands

```bash
# Model loading test (works)
timeout 15s ./vibrant --device cuda ask --model qwen2.5-coder-3b-q4 "test"
# Should see: "Model loaded on GPU" before timeout

# GPU memory check (works)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# Should show ~12GB after model load

# Inference test (hangs)
./vibrant --device cuda ask --model qwen2.5-coder-3b-q4 "What is 2+2?"
# Hangs at "Generating response..."
```

##Related Files

- CUDA kernels: `internal/gpu/cuda/kernels.cu`
- Kernel wrappers: `internal/gpu/cuda/library.go`
- Tensor ops: `internal/tensor/ops_cuda.go`
- GPU device: `internal/gpu/cuda.go`

## Recommendation

The dequantization feature should be **documented but marked as experimental** until the inference hang is resolved. The core functionality (dequantization + GPU transfer) is working, but the CUDA kernel execution needs debugging.

**Estimated effort to fix**: 4-8 hours of CUDA-specific debugging with proper logging and kernel verification.
