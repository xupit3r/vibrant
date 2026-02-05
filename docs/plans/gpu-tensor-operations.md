# GPU Tensor Operations - Implementation Plan

**Goal:** Make all tensor operations device-aware to keep intermediate tensors on GPU throughout the forward pass, achieving full GPU acceleration.

**Target:** 70-95% GPU utilization, ~10x inference speedup

## Problem Statement

Currently, tensor operations (RMSNorm, RoPE, Softmax, Add, etc.) create new tensors on CPU by default, even when input tensors are on GPU. This forces expensive CPU↔GPU data transfers and prevents GPU matmul acceleration.

Example of current behavior:
```
Input:  hidden [GPU] 
   ↓
RMSNorm: normalized = NewTensor(...) → [CPU]  ❌
   ↓
MatMul:  Can't use GPU (input is CPU) → Fallback to CPU
```

Desired behavior:
```
Input:  hidden [GPU]
   ↓
RMSNorm: normalized = NewTensorOnDevice(hidden.Device()) → [GPU] ✅
   ↓
MatMul:  Both tensors on GPU → Use GPU kernel 🚀
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Priority 1)

#### 1.1 Add Device-Aware Tensor Creation
**File:** `internal/tensor/tensor.go`

```go
// NewTensorOnDevice creates a tensor on the specified device
func NewTensorOnDevice(shape []int, dtype DType, device Device) *Tensor {
    t := &Tensor{
        shape:  shape,
        dtype:  dtype,
        device: device,
        stride: computeStride(shape),
    }
    
    if device == GPU {
        // Allocate directly on GPU
        size := calculateSize(shape, dtype)
        dev, _ := gpu.GetDefaultDevice()
        buf, _ := dev.Allocate(size)
        t.gpuBuffer = buf
        t.gpuDevice = dev
        // Don't allocate CPU backing store
        t.data = nil
    } else {
        // Existing CPU allocation
        t.data = allocateData(shape, dtype)
    }
    
    return t
}

// Clone creates a copy of tensor on the same device
func (t *Tensor) Clone() *Tensor {
    cloned := NewTensorOnDevice(t.shape, t.dtype, t.device)
    // Copy data (GPU→GPU or CPU→CPU)
    if t.device == GPU {
        t.gpuDevice.Copy(cloned.gpuBuffer, t.gpuBuffer, t.gpuBuffer.Size())
    } else {
        copy(cloned.Data().([]float32), t.Data().([]float32))
    }
    return cloned
}
```

**Testing:**
- Create tensor on CPU, verify data is on CPU
- Create tensor on GPU, verify gpuBuffer is allocated
- Clone GPU tensor, verify copy stays on GPU

#### 1.2 Add GPU Kernels for Common Operations
**Files:** 
- `internal/gpu/cuda/kernels.cu` - CUDA implementations
- `internal/gpu/cuda/kernels.go` - Go wrappers

Operations to implement:
```cuda
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
__global__ void rmsNormKernel(float* out, const float* in, 
                                const float* weight, int N, float eps);

// Element-wise operations
__global__ void addKernel(float* out, const float* a, const float* b, int N);
__global__ void mulKernel(float* out, const float* a, const float* b, int N);
__global__ void siluKernel(float* out, const float* in, int N);  // x * sigmoid(x)

// Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
__global__ void softmaxKernel(float* out, const float* in, int rows, int cols);

// RoPE: Apply rotary position embeddings
__global__ void ropeKernel(float* out, const float* in, const float* cos, 
                            const float* sin, int seq_len, int head_dim);
```

Go wrapper pattern:
```go
func (k *KernelSet) RMSNorm(out, in, weight Buffer, N int, eps float32) error {
    // Calculate grid/block dimensions
    blockSize := 256
    gridSize := (N + blockSize - 1) / blockSize
    
    // Launch kernel
    return k.launchRMSNorm(
        unsafe.Pointer(out.Ptr()),
        unsafe.Pointer(in.Ptr()),
        unsafe.Pointer(weight.Ptr()),
        N, eps,
        gridSize, blockSize,
    )
}
```

### Phase 2: Update Tensor Operations (Priority 1)

#### 2.1 RMSNorm
**File:** `internal/transformer/norm.go`
**Current:** Creates CPU tensor for output
**Fix:**

```go
func (r *RMSNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
    shape := x.Shape()
    hiddenDim := shape[len(shape)-1]
    
    // Create output on same device as input
    output := tensor.NewTensorOnDevice(shape, tensor.Float32, x.Device())
    
    if x.IsOnGPU() {
        // GPU path
        kernels := getKernelsForDevice(x.GPUDevice())
        err := kernels.RMSNorm(
            output.GPUBuffer(),
            x.GPUBuffer(),
            r.weight.GPUBuffer(),
            tensor.NumElements(shape),
            r.eps,
        )
        if err != nil {
            return nil, err
        }
        return output, nil
    }
    
    // CPU path (existing implementation)
    // ... normalize on CPU ...
    
    return output, nil
}
```

**Testing:**
- CPU input → CPU output with correct normalization
- GPU input → GPU output with correct normalization
- Compare CPU vs GPU outputs (should match within tolerance)

#### 2.2 RoPE (Rotary Position Embeddings)
**File:** `internal/tensor/rope.go`
**Current:** Creates CPU tensors for rotated Q/K
**Fix:**

```go
func ApplyRotaryEmbeddings(q, k *tensor.Tensor, positions []int, base float64) (*tensor.Tensor, *tensor.Tensor, error) {
    // Pre-compute cos/sin on same device
    cosCache, sinCache := computeRoPECache(positions, q.Shape(), base, q.Device())
    
    // Create outputs on same device
    qRot := tensor.NewTensorOnDevice(q.Shape(), tensor.Float32, q.Device())
    kRot := tensor.NewTensorOnDevice(k.Shape(), tensor.Float32, k.Device())
    
    if q.IsOnGPU() {
        // GPU path
        kernels := getKernelsForDevice(q.GPUDevice())
        err := kernels.RoPE(qRot.GPUBuffer(), q.GPUBuffer(), 
                            cosCache.GPUBuffer(), sinCache.GPUBuffer(), ...)
        // ... similar for k ...
        return qRot, kRot, nil
    }
    
    // CPU path (existing)
    // ...
}
```

#### 2.3 Softmax
**File:** `internal/tensor/softmax.go`
**Current:** Creates CPU tensor for probabilities
**Fix:**

```go
func Softmax(x *tensor.Tensor, axis int) *tensor.Tensor {
    // Create output on same device
    output := tensor.NewTensorOnDevice(x.Shape(), tensor.Float32, x.Device())
    
    if x.IsOnGPU() {
        // GPU path - numerically stable softmax
        kernels := getKernelsForDevice(x.GPUDevice())
        rows, cols := computeRowsCols(x.Shape(), axis)
        err := kernels.Softmax(output.GPUBuffer(), x.GPUBuffer(), rows, cols)
        if err != nil {
            // Fallback to CPU
            return softmaxCPU(x)
        }
        return output
    }
    
    // CPU path
    return softmaxCPU(x)
}
```

#### 2.4 Element-wise Operations
**File:** `internal/tensor/ops.go`
**Current:** `Add()`, `Mul()`, `SiLU()` create CPU tensors
**Fix:**

```go
func Add(a, b *tensor.Tensor) *tensor.Tensor {
    // Verify same device
    if a.Device() != b.Device() {
        // Move b to a's device if needed
        b, _ = b.ToDevice(a.Device())
    }
    
    result := tensor.NewTensorOnDevice(a.Shape(), a.DType(), a.Device())
    
    if a.IsOnGPU() {
        kernels := getKernelsForDevice(a.GPUDevice())
        err := kernels.Add(result.GPUBuffer(), a.GPUBuffer(), b.GPUBuffer(), 
                           tensor.NumElements(a.Shape()))
        if err == nil {
            return result
        }
    }
    
    // CPU fallback
    return addCPU(a, b)
}

// Similar for Mul, Sub, Div, SiLU, etc.
```

### Phase 3: Attention Optimizations (Priority 2)

#### 3.1 Fused Attention Operations
Combine multiple operations to reduce memory traffic:

```cuda
// Fused QK^T + Softmax
__global__ void fusedQKSoftmaxKernel(
    float* attn,        // output: [batch, heads, seq, seq]
    const float* Q,     // [batch, heads, seq, head_dim]
    const float* K,     // [batch, heads, seq, head_dim]
    int seq_len,
    int head_dim,
    float scale
) {
    // Compute QK^T and softmax in a single kernel
    // Reduces intermediate storage and memory bandwidth
}
```

#### 3.2 Attention Score Calculation
**File:** `internal/transformer/attention.go`

Current flow:
```
Q, K, V [GPU] → MatMul(Q, K^T) [CPU] → Softmax [CPU] → MatMul(attn, V) [CPU]
```

Target flow:
```
Q, K, V [GPU] → MatMul(Q, K^T) [GPU] → Softmax [GPU] → MatMul(attn, V) [GPU]
```

### Phase 4: FeedForward Network (Priority 2)

**File:** `internal/transformer/feedforward.go`

Current bottleneck:
```go
func (f *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
    gate := tensor.MatMul(x, f.gate)    // [CPU] even if x on GPU
    up := tensor.MatMul(x, f.up)        // [CPU]
    gateActivated := tensor.SiLU(gate)  // [CPU]
    combined := tensor.Mul(gateActivated, up)  // [CPU]
    output := tensor.MatMul(combined, f.down)  // [CPU]
    return output
}
```

Target:
```go
func (f *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
    // All operations stay on GPU if x is on GPU
    gate := tensor.MatMul(x, f.gate)              // [GPU]
    up := tensor.MatMul(x, f.up)                  // [GPU]
    gateActivated := tensor.SiLU(gate)            // [GPU] - new GPU kernel
    combined := tensor.Mul(gateActivated, up)     // [GPU] - new GPU kernel
    output := tensor.MatMul(combined, f.down)     // [GPU]
    return output
}
```

## Implementation Order

### Week 1: Core Infrastructure
- [x] Review existing CUDA infrastructure
- [ ] Implement `NewTensorOnDevice()` and device-aware creation
- [ ] Add basic GPU kernels (Add, Mul, SiLU)
- [ ] Write unit tests for GPU kernels
- [ ] Verify kernel correctness vs CPU

### Week 2: Critical Operations
- [ ] Implement GPU RMSNorm kernel
- [ ] Update `RMSNorm.Forward()` to use GPU kernel
- [ ] Implement GPU Softmax kernel
- [ ] Update `Softmax()` to use GPU kernel
- [ ] Verify normalization and softmax outputs match CPU

### Week 3: Positional Embeddings & Attention
- [ ] Implement GPU RoPE kernel
- [ ] Update `ApplyRotaryEmbeddings()` to use GPU
- [ ] Update attention score calculation for GPU
- [ ] Test end-to-end attention with GPU

### Week 4: FeedForward & Integration
- [ ] Update FeedForward to preserve GPU tensors
- [ ] Add device-awareness to all remaining ops
- [ ] Full integration testing
- [ ] Performance benchmarking

## Testing Strategy

### Unit Tests
Each GPU operation needs:
1. **Correctness test:** GPU output matches CPU output
2. **Device test:** Output is on correct device
3. **Shape test:** Output has expected shape
4. **Edge cases:** Empty tensors, single elements, large tensors

Example:
```go
func TestRMSNormGPU(t *testing.T) {
    if !tensor.HasGPU() {
        t.Skip("GPU not available")
    }
    
    // Create test input
    x := tensor.NewTensor([]int{2, 4, 8}, tensor.Float32)
    fillRandom(x)
    
    // CPU reference
    cpuNorm := RMSNormCPU(x, eps)
    
    // GPU test
    xGPU, _ := x.ToDevice(tensor.GPU)
    gpuNorm := RMSNormGPU(xGPU, eps)
    gpuNormCPU, _ := gpuNorm.ToDevice(tensor.CPU)
    
    // Compare
    assertClose(t, cpuNorm, gpuNormCPU, 1e-5)
    assert.True(t, gpuNorm.IsOnGPU())
}
```

### Integration Tests
Full forward pass tests:
```go
func TestForwardPassGPU(t *testing.T) {
    model := loadTestModel()
    model.MoveToDevice(tensor.GPU)
    
    input := [][]int{{1, 2, 3, 4}}
    
    // Run forward pass
    logits, err := model.Forward(input, false)
    
    // Verify output
    assert.NoError(t, err)
    assert.True(t, logits.IsOnGPU())
    
    // Verify GPU was actually used
    // (check nvidia-smi showed >50% utilization)
}
```

### Performance Benchmarks
```go
func BenchmarkInferenceCPU(b *testing.B) {
    model := loadModel()
    input := [][]int{{1, 2, 3, 4, 5}}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        model.Forward(input, false)
    }
}

func BenchmarkInferenceGPU(b *testing.B) {
    model := loadModel()
    model.MoveToDevice(tensor.GPU)
    input := [][]int{{1, 2, 3, 4, 5}}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        model.Forward(input, false)
    }
}
```

## Success Criteria

### Functional
- ✅ All tensor operations preserve input device
- ✅ GPU kernels produce same output as CPU (within 1e-5)
- ✅ No unexpected device transfers during forward pass
- ✅ Model generates correct text on GPU

### Performance
- ✅ GPU utilization >70% during inference
- ✅ Inference speed 8-10x faster than CPU
- ✅ First token latency <1 second
- ✅ Tokens/second >50 for 3B model on RTX 4090

### Observability
- Monitor with: `watch -n 0.5 nvidia-smi`
- Expected: Memory stable at 13GB, GPU util 70-95%
- No CPU→GPU transfer spikes during steady state

## Risks & Mitigations

### Risk 1: Numerical Instability
**Problem:** GPU floating-point arithmetic can differ from CPU
**Mitigation:** 
- Use same algorithm as CPU (e.g., numerically stable softmax)
- Test with multiple random seeds
- Accept tolerance of 1e-5 for Float32

### Risk 2: Memory Overhead
**Problem:** Keeping all tensors on GPU uses more VRAM
**Mitigation:**
- Reuse buffers where possible
- Free intermediate tensors aggressively
- Fall back to CPU if OOM

### Risk 3: Complexity
**Problem:** Maintaining separate CPU/GPU code paths
**Mitigation:**
- Share algorithm logic, only dispatch changes
- Extensive testing to catch divergence
- Clear documentation of which path is used when

## Future Optimizations

After Phase 1-4 are complete:

1. **Kernel Fusion:** Combine operations to reduce memory traffic
2. **Flash Attention:** Memory-efficient attention implementation
3. **Quantized Kernels:** Native Q4_K GPU operations
4. **Mixed Precision:** FP16/BF16 for faster computation
5. **Multi-GPU:** Distribute model across GPUs
6. **Persistent Kernels:** Reduce kernel launch overhead

## References

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch GPU kernels: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda
- Flash Attention paper: https://arxiv.org/abs/2205.14135
- Existing implementation: `internal/gpu/cuda/kernels.cu`
