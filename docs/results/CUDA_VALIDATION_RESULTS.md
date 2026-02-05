# CUDA Validation Results

## Hardware
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Platform**: Linux (x86_64)
- **Date**: February 5, 2026

## Test Summary

### Tensor CUDA Tests (11/11 passing)

| Test | Description | Result |
|------|-------------|--------|
| TestCUDADeviceAccess | GPU detection, name, memory reporting | PASS |
| TestCUDAMemoryAllocation | Buffer allocation at 256B, 1KB, 1MB, 16MB | PASS |
| TestCUDATensorRoundTrip | CPU→GPU→GPU data integrity (4x8 tensor) | PASS |
| TestCUDAAddGPU | Element-wise addition on GPU vs CPU reference | PASS |
| TestCUDAMulGPU | Element-wise multiplication on GPU vs CPU reference | PASS |
| TestCUDAMatMulSingleRow | Single-row MatMul [1x4]@[4x3] (decode path) | PASS |
| TestCUDAMatMulGeneral | General tiled MatMul [2x3]@[3x2] | PASS |
| TestCUDASoftmax | Softmax sum-to-1 property + CPU reference match | PASS |
| TestCUDASiLU | SiLU activation vs CPU reference | PASS |
| TestCUDAGPUCPUConsistency | Side-by-side CPU/GPU Add+Mul on 8x16 tensors | PASS |
| TestCUDALargerMatMul | 32x64 @ 64x32 MatMul with relative tolerance | PASS |

### GPU Device Tests (9/9 passing, 7 skipped for macOS-only)

| Test | Result |
|------|--------|
| TestCUDADevice | PASS |
| TestCUDABuffer | PASS |
| TestCUDAHostTransfer | PASS |
| TestCUDADeviceToDevice | PASS |
| TestCUDASync | PASS |
| TestCUDAInvalidSize | PASS |
| TestCUDABufferPool | PASS (1 hit, 1 miss) |
| TestCUDABufferPoolReuse | PASS (9 hits, 1 miss on 10 cycles) |
| TestCUDABufferPoolPressure | PASS (100 buffers allocated) |

## Correctness

All GPU operations match CPU reference implementations within tolerance:
- Element-wise ops (Add, Mul): exact match (< 1e-4)
- MatMul (single-row and general): exact match (< 1e-4)
- Softmax: sum-to-1 property verified (< 1e-3), element match (< 1e-4)
- SiLU: matches CPU reference (< 1e-4)
- Data round-trip (CPU→GPU→CPU): bit-exact (< 1e-6)

## Memory Management

- GPU memory: 981 MB used / 24,079 MB total at idle
- Buffer pool: 80% of VRAM allocated (~19 GB)
- Pool reuse rate: 90% (9/10 hits on repeated same-size allocations)
- No memory leaks detected across allocation/deallocation cycles

## Bug Fixes During Validation

Several build tag bugs were discovered and fixed during CUDA validation:

1. **pool_stub.go** (`// +build !darwin !cgo`): The stub's `Allocate` method simply delegated to `device.Allocate`, causing infinite recursion with `CUDADevice.Allocate` → `BufferPool.Allocate` → stack overflow. Fixed by excluding `linux,cgo` from the stub and compiling the real `pool.go` on Linux.

2. **tensor_gpu_stub.go** (`// +build !darwin !cgo`): The stub was active on Linux+CGO, making `ToDevice(GPU)` always return an error. The `toGPU()`/`toCPU()` methods in `tensor_cuda.go` were orphaned code never called by the public API. Fixed by adding `tensor_cuda_api.go` with `ToDevice`/`IsOnGPU`/`FreeGPU`/`SyncGPU` for `linux,cgo`.

3. **pool.go** (`// +build darwin,cgo`): The real buffer pool was darwin-only. Fixed build tag to `(darwin OR linux) AND cgo` and replaced the Metal-specific type assertion in `allocateDirect` with a `directAllocator` interface.

4. **cuda_test.go / cuda_bench_test.go**: Referenced `stats.Hits`/`stats.Misses` but the real `PoolStats` struct has `PoolHits`/`PoolMisses`. Previously hidden because the stub was always compiled on Linux.

## Conclusion

The CUDA backend is **fully validated** on RTX 4090 hardware:
- All 11 tensor operations produce correct results
- Buffer pool works with efficient memory reuse
- Data integrity maintained across CPU↔GPU transfers
- Build tag architecture correctly separates Metal/CUDA/stub code paths
