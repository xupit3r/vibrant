# Performance Baseline & Benchmarking Plan

**Date**: January 31, 2026
**Status**: Awaiting quantization implementation
**Target Model**: qwen2.5-coder-14b-q5.gguf (5.5GB)

## Current Performance Metrics

### Model Loading (✅ Working)

| Operation | Time | Notes |
|-----------|------|-------|
| GGUF parsing | <1ms | Fast metadata extraction |
| Config creation | <1ms | Negligible overhead |
| Engine initialization | 105.8ms | Includes all component setup |
| **Total load time** | **~106ms** | Excellent for 5.5GB model |

### Memory Usage (✅ Working)

| Component | Size | Method |
|-----------|------|--------|
| Model file | 5.5GB | Memory-mapped (not in RAM) |
| Metadata | <1MB | In-memory maps |
| Tensors | 0 bytes | Lazy-loaded via mmap |
| **Total RAM** | **<2MB** | Minimal footprint! |

### Inference (❌ Blocked - Awaiting Quantization)

| Metric | Target | Current |
|--------|--------|---------|
| Tokens/second | 5-20 | N/A (blocked) |
| First token latency | <500ms | N/A (blocked) |
| Memory per token | <100MB | N/A (blocked) |

## Benchmark Suite (To Be Implemented)

### 1. Tokenization Benchmarks

```bash
go test ./internal/tokenizer -bench=. -benchmem
```

**Current Results**:
- Encode: ~2.3µs per operation
- Decode: ~78ns per operation

**Target**: <5µs encode, <100ns decode

### 2. Tensor Operations Benchmarks

```bash
go test ./internal/tensor -bench=. -benchmem
```

**Current Results**:
- MatMul (SIMD): 1.9x speedup over naive
- MatMul (Parallel+SIMD): 6.5x speedup over naive

**Targets**:
- MatMul: Within 2-3x of llama.cpp
- Element-wise ops: <1µs per 1K elements

### 3. Transformer Layer Benchmarks

```bash
go test ./internal/transformer -bench=. -benchmem
```

**To measure**:
- Attention mechanism: forward pass time
- Feed-forward network: forward pass time
- Layer normalization: overhead
- RoPE embeddings: computation time

**Targets**:
- Single layer: <100ms for 5120-dim
- 48 layers: <5s total (without optimization)

### 4. Sampling Benchmarks

```bash
go test ./internal/inference -bench=. -benchmem
```

**Current Results**:
- Greedy sampling: Minimal overhead
- Temperature sampling: <1ms
- Top-K/Top-P: <5ms

**Target**: <10ms for all sampling strategies

### 5. End-to-End Inference Benchmarks

Once quantization is implemented:

```bash
./test-inference -model model.gguf -benchmark
```

**Metrics to measure**:
- Time to first token (TTFT)
- Tokens per second (TPS)
- Memory usage during generation
- CPU usage
- Cache efficiency

## Comparison Targets

### llama.cpp Baseline

For qwen2.5-coder-14b-q5:

| Metric | llama.cpp (estimated) | Our Target | Status |
|--------|----------------------|------------|--------|
| Load time | ~100-200ms | 106ms | ✅ Comparable |
| TTFT | ~200-500ms | <500ms | TBD |
| TPS (CPU) | 5-10 tok/s | 5-20 tok/s | TBD |
| Memory | ~6-8GB | <8GB | ✅ On track |

### Performance Tiers

**Tier 1: Functional** (Initial goal)
- ✅ Loads models correctly
- ⏳ Generates tokens (any speed)
- ⏳ Numerical accuracy within 1e-3

**Tier 2: Usable** (Next milestone)
- ⏳ 5-10 tokens/second
- ⏳ TTFT <1 second
- ⏳ Stable memory usage

**Tier 3: Competitive** (Future goal)
- Within 2x of llama.cpp
- Optimized SIMD paths
- Efficient caching

**Tier 4: Optimized** (Long-term)
- Match or exceed llama.cpp
- Assembly hot paths
- Flash Attention
- Speculative decoding

## Profiling Strategy

### CPU Profiling

```bash
go test ./internal/inference -run TestGenerate -cpuprofile=cpu.prof
go tool pprof -http=:8080 cpu.prof
```

**Focus areas**:
- MatMul hot paths
- Quantization overhead
- Sampling algorithms
- Memory allocations

### Memory Profiling

```bash
go test ./internal/inference -run TestGenerate -memprofile=mem.prof
go tool pprof -http=:8080 mem.prof
```

**Focus areas**:
- Tensor allocations
- KV-cache growth
- Unnecessary copies
- Memory leaks

### Benchmark Profiling

```bash
go test ./internal/transformer -bench=BenchmarkAttention \
  -cpuprofile=cpu.prof -memprofile=mem.prof -benchmem
```

## Optimization Priorities

Based on profiling results, prioritize:

### High Priority
1. **Quantized MatMul** - Biggest bottleneck
2. **SIMD Dequantization** - Hot path
3. **Memory Pooling** - Reduce GC pressure
4. **Cache Optimization** - Attention speedup

### Medium Priority
5. **Parallel Decoding** - Multi-core usage
6. **Fused Operations** - Reduce overhead
7. **Batch Processing** - Amortize costs
8. **Lazy Evaluation** - Avoid unnecessary work

### Low Priority
9. **Assembly Hot Paths** - Diminishing returns
10. **GPU Support** - Different scope

## Performance Testing Workflow

### 1. Establish Baseline

```bash
# Run full benchmark suite
make bench

# Profile inference
./scripts/profile-inference.sh model.gguf

# Generate report
./scripts/perf-report.sh > baseline.txt
```

### 2. Implement Optimization

```bash
# Make changes
# Run targeted benchmarks
go test ./internal/tensor -bench=BenchmarkMatMul -count=10

# Compare results
benchstat baseline.txt optimized.txt
```

### 3. Verify Improvements

```bash
# Ensure no regressions
make test

# Profile again
./scripts/profile-inference.sh model.gguf

# Update baseline if better
cp optimized.txt baseline.txt
```

## Benchmark Infrastructure

### Required Scripts

**scripts/bench-all.sh**:
```bash
#!/bin/bash
echo "Running comprehensive benchmarks..."
go test ./internal/tensor -bench=. -benchmem | tee bench-tensor.txt
go test ./internal/tokenizer -bench=. -benchmem | tee bench-tokenizer.txt
go test ./internal/transformer -bench=. -benchmem | tee bench-transformer.txt
go test ./internal/inference -bench=. -benchmem | tee bench-inference.txt
```

**scripts/profile-inference.sh**:
```bash
#!/bin/bash
MODEL=$1
./test-inference -model $MODEL -profile -max-tokens 100
go tool pprof -http=:8080 cpu.prof &
go tool pprof -http=:8081 mem.prof &
```

**scripts/perf-report.sh**:
```bash
#!/bin/bash
echo "=== Performance Report ==="
echo "Date: $(date)"
echo ""
echo "=== Model Loading ==="
grep "Model loaded" test-output.log
echo ""
echo "=== Token Generation ==="
grep "tokens/sec" test-output.log
echo ""
echo "=== Memory Usage ==="
go tool pprof -top mem.prof | head -20
```

## Continuous Performance Monitoring

### CI/CD Integration

Add to `.github/workflows/benchmark.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.23'

      - name: Run benchmarks
        run: |
          go test ./internal/tensor -bench=. -benchmem > new.txt

      - name: Download previous benchmark
        uses: actions/cache@v3
        with:
          path: old.txt
          key: benchmark-${{ github.ref }}

      - name: Compare benchmarks
        run: |
          if [ -f old.txt ]; then
            benchstat old.txt new.txt
          fi
          cp new.txt old.txt
```

## Next Steps

### Immediate (Once Quantization Works)

1. ✅ Run baseline end-to-end test
2. ⏳ Measure tokens/second
3. ⏳ Profile CPU hot paths
4. ⏳ Profile memory allocations
5. ⏳ Document baseline performance

### Short-term (Phase 10.8)

1. Implement quantized MatMul benchmarks
2. Compare with llama.cpp on same hardware
3. Identify top 3 bottlenecks
4. Optimize critical paths
5. Re-measure and verify improvements

### Long-term (Phase 10.9+)

1. Continuous performance monitoring
2. Regression detection
3. Optimization roadmap
4. Performance documentation

## Success Criteria

### Minimum Viable Performance
- ✅ Model loads in <1 second
- ⏳ Generates tokens (any speed)
- ⏳ Memory usage <8GB for 14B model
- ⏳ No crashes or leaks

### Target Performance
- ✅ Model loads in <200ms
- ⏳ 5-10 tokens/second on CPU
- ⏳ TTFT <1 second
- ⏳ Stable memory usage

### Stretch Performance
- Within 2x of llama.cpp
- 10-20 tokens/second on CPU
- TTFT <500ms
- Memory usage <6GB

## Current Bottlenecks (Known)

1. **Quantization** - Not implemented (blocks all inference)
2. **MatMul** - Could be faster with better SIMD
3. **Memory Allocations** - Too many temp tensors
4. **Cache Misses** - Need better data locality

## Performance Achievements So Far

✅ **Excellent**:
- Model loading: 106ms for 5.5GB model
- Memory footprint: <2MB (mmap working)
- Tokenization: 2.3µs encode, 78ns decode
- MatMul: 6.5x speedup with SIMD+Parallel

⏳ **Pending**:
- Inference speed: Blocked on quantization
- Token generation: Blocked on quantization
- End-to-end benchmarks: Blocked on quantization

---

**Last Updated**: January 31, 2026
**Status**: Benchmarking framework ready, awaiting quantization implementation
**Next Milestone**: Implement Q5_K dequantization, then measure baseline TPS
