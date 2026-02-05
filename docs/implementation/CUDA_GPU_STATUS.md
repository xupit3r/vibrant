# CUDA GPU Support - Implementation Status

**Last Updated:** 2026-02-05  
**Status:** Partially Working - Model loads to GPU, inference optimization in progress

## Overview

CUDA GPU support has been successfully implemented for Vibrant. The model loads to GPU memory and CUDA kernels are functional, but full GPU acceleration requires making all tensor operations device-aware to keep intermediate tensors on GPU throughout the forward pass.

## ✅ What's Working

### Core Infrastructure
- ✅ CUDA device detection and initialization
- ✅ Device flag recognition (`--device cuda`, `--device gpu`)
- ✅ CUDA memory management (buffer pool with 80% GPU memory)
- ✅ GPU memory transfers (cudaMalloc, cudaMemcpy)
- ✅ CUDA kernel compilation and execution
- ✅ GPU synchronization primitives

### Model Loading
- ✅ Quantized model loading (Q4_K_M, Q5_K_M, Q6_K)
- ✅ Automatic dequantization to Float32 for GPU
- ✅ Model weight transfer to GPU (~13GB VRAM for 3B model)
- ✅ Large tensor support (>1GB tensors handled correctly)
- ✅ Embedding weight management

### Memory Management
- ✅ Buffer pooling (prevents fragmentation)
- ✅ Buffer reuse for efficiency
- ✅ Proper cleanup and deallocation
- ✅ OOM handling and graceful fallback

### Verification
```bash
# GPU memory usage observed:
- Startup: ~500MB (system)
- Model load: 500MB → 13GB (model weights + activations)
- Inference: 13GB stable
- GPU utilization: 1-17% (limited by CPU tensor ops)
```

## ⚠️ Partially Working

### GPU Inference
**Status:** Model runs but most operations fall back to CPU

**Root Cause:** Intermediate tensors are created on CPU during forward pass, forcing MatMul operations to use CPU fallback even though model weights are on GPU.

**Symptoms:**
- Low GPU utilization (1-17% instead of 70-100%)
- Slow inference (comparable to CPU-only)
- Debug logs show: `MatMul: Not using GPU (a.IsOnGPU=false, b.IsOnGPU=true)`

**Impact:** Inference works but doesn't get the expected ~10x speedup from GPU acceleration.

## 🔧 Remaining Work

### Phase 1: Device-Aware Tensor Operations (Current)
Make all tensor operations preserve the device of input tensors:

#### High Priority (Block GPU matmul)
1. **RMSNorm** - `internal/transformer/norm.go`
   - Forward() creates new CPU tensor for normalized output
   - Need to detect input device and create output on same device

2. **RoPE (Rotary Position Embeddings)** - `internal/tensor/rope.go`
   - applyRotaryEmbeddings() creates CPU tensors for rotated Q/K
   - Need device-aware tensor creation

3. **Softmax** - `internal/tensor/softmax.go`
   - Creates CPU output tensor
   - Need to preserve input device

4. **Element-wise ops** - `internal/tensor/ops.go`
   - Add(), Mul(), etc. create CPU tensors
   - Need device detection and GPU implementations

#### Medium Priority (Improve GPU efficiency)
5. **Attention** - `internal/transformer/attention.go`
   - Intermediate QK^T, attention scores on CPU
   - Need device-aware intermediate storage

6. **FeedForward** - `internal/transformer/feedforward.go`
   - SiLU activation creates CPU tensor
   - Gate * Up multiplication creates CPU result

7. **Reshape/View operations**
   - Should preserve device metadata
   - Currently force CPU copy

#### Low Priority (Optimize later)
8. **Quantized operations** - GPU-native quantized kernels
9. **Fused kernels** - Combine multiple ops (e.g., MatMul + ReLU)
10. **Multi-GPU support** - Distribute across multiple GPUs

### Phase 2: Performance Optimization
- Kernel fusion (combine operations to reduce memory transfers)
- Stream scheduling (overlap compute and memory transfers)
- Memory pooling tuning
- Profiling and bottleneck identification

### Phase 3: Advanced Features
- Flash Attention (memory-efficient attention)
- Native quantized inference (avoid dequantization overhead)
- Mixed precision training/inference (FP16/BF16)

## Technical Architecture

### Memory Flow
```
CPU: Load GGUF → Dequantize Q4_K → Float32 weights
                        ↓
GPU: cudaMemcpy → Buffer Pool → Model Weights (13GB)
                        ↓
CPU: Embed Tokens → []float32 embeddings
                        ↓
GPU: cudaMemcpy → Embeddings on GPU
                        ↓
      [Forward Pass - Mixed CPU/GPU]
                        ↓
CPU: Sample Next Token ← cudaMemcpy ← Logits from GPU
```

**Current Bottleneck:** Intermediate tensors created on CPU force data transfer back and forth.

### Device Tracking
Each tensor tracks its device via:
- `tensor.device` field (CPU or GPU enum)
- `tensor.gpuBuffer` pointer (non-nil if on GPU)
- `tensor.IsOnGPU()` method for checking

Model tracks target device:
- `model.device` field set by `MoveToDevice()`
- Used to determine if embeddings should be moved to GPU
- Should be propagated to all tensor operations

### Buffer Management
- **Buffer Pool:** 80% of GPU memory (~19GB on RTX 4090)
- **Strategy:** Round sizes to power-of-2 for reuse
- **Allocation:** Try pool first, fall back to direct cudaMalloc
- **Deallocation:** Return to pool up to maxBytes limit

## Build & Test

### Building
```bash
# Build with CUDA support
make build-cuda

# This compiles CUDA kernels and links with CGO_ENABLED=1
# Output: ./vibrant (binary with CUDA support)
```

### Testing
```bash
# Run with CUDA
export LD_LIBRARY_PATH=./build/cuda:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
./vibrant --device cuda ask --model qwen2.5-coder-3b-q4 "test"

# Or use wrapper script
./vibrant-cuda.sh --device cuda chat
```

### Monitoring
```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run Vibrant
./vibrant-cuda.sh --device cuda ask "What is 2+2?"

# Expected output:
# - Memory: ~500MB → ~13GB during model load
# - Utilization: 1-17% during inference (will improve with Phase 1)
```

## Performance Metrics

### Current (Partial GPU)
- **Model Load Time:** ~15 seconds (includes dequantization)
- **First Token:** ~2-3 seconds
- **Tokens/sec:** ~5-10 (limited by CPU tensor ops)
- **GPU Utilization:** 1-17%
- **VRAM Usage:** 13GB

### Target (Full GPU) - After Phase 1
- **Model Load Time:** ~10 seconds
- **First Token:** ~0.5 seconds  
- **Tokens/sec:** ~50-80 (10x improvement)
- **GPU Utilization:** 70-95%
- **VRAM Usage:** 13GB (same)

## Known Issues

1. **Embedding lookups on CPU** (by design)
   - Random access patterns are faster on CPU
   - Solution: Transfer embedding output to GPU (already implemented)

2. **Intermediate tensors on CPU** (blocking full GPU acceleration)
   - All tensor ops create CPU output by default
   - Solution: Device-aware tensor creation (Phase 1 work)

3. **No quantized GPU kernels** (memory inefficient)
   - Dequantize Q4_K → Float32 → 4x memory overhead
   - Solution: Native Q4_K CUDA kernels (Phase 3)

4. **Buffer pool disabled temporarily during debugging**
   - Re-enabled now, working correctly
   - No known issues

## Dependencies

### Required
- CUDA Toolkit 11.0+ (tested with 13.1)
- NVIDIA GPU with Compute Capability 6.0+ (tested on RTX 4090)
- GCC/G++ for CGO compilation
- Go 1.17+ (for unsafe.Slice)

### Build Files
- `internal/gpu/cuda/*.cu` - CUDA kernel implementations
- `internal/gpu/cuda/*.h` - CUDA headers  
- `Makefile` - `build-cuda` target compiles kernels
- `build/cuda/` - Compiled `.so` library output

## References

### Documentation
- `specs/cuda-backend.md` - CUDA backend specification
- `docs/implementation/GPU_QUANTIZATION_SUPPORT.md` - Quantization details
- `README.md` - User-facing GPU setup instructions

### Code Entry Points
- `internal/llm/engine_custom.go` - Device flag handling
- `internal/tensor/tensor_cuda.go` - GPU tensor operations
- `internal/gpu/cuda.go` - CUDA device management
- `internal/transformer/model.go` - Model device migration

## Next Steps

See `docs/plans/gpu-tensor-operations.md` for detailed implementation plan for Phase 1.

Priority: Make tensor operations device-aware to achieve full GPU acceleration.
