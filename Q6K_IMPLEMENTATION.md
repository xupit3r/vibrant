# Q6_K Quantization Implementation

## Summary

Implemented complete Q6_K quantized tensor support for the Vibrant LLM inference engine, enabling 6-bit quantized model weights with 6.5625 bits per weight compression ratio.

## Implementation Details

### Files Created

1. **`internal/tensor/quant_q6k.go`** - Core Q6_K quantization/dequantization
   - `BlockQ6_K` struct (210 bytes per 256 values)
   - `DequantizeQ6_K()` - Batch dequantization
   - `DequantizeQ6_KElement()` - Single element access
   - `DequantizeQ6_KTensor()` - Full tensor conversion
   - `QuantizeQ6_K()` - Quantization for testing
   - `parseQ6_KBlock()` - Block parsing from raw bytes

2. **`internal/tensor/quant_q6k_test.go`** - Comprehensive test suite
   - 15 unit tests covering all functionality
   - Roundtrip quantization tests
   - Edge case handling (out of bounds, size mismatches)
   - Bit packing verification
   - 4 benchmarks for performance measurement

### Files Modified

3. **`internal/tensor/tensor.go`**
   - Added `Q6_K` to `DataType` enum
   - Updated `String()` method to return "q6_k"
   - Updated `BytesPerElement()` to handle Q6_K
   - Updated `NewTensor()` to allocate Q6_K data
   - Updated `NewTensorMmap()` to handle Q6_K
   - Updated `At()` method to dequantize Q6_K elements

4. **`internal/tensor/matmul.go`**
   - Added Q6_K dequantization before matrix multiplication
   - Handles both A and B matrices as Q6_K

5. **`internal/gguf/metadata.go`**
   - Updated `ggmlTypeToTensorType()` to map `GGML_TYPE_Q6_K` to `tensor.Q6_K`
   - Q6_K constant already existed (value 14)
   - Size calculation already implemented in `calculateTensorSize()`

6. **`internal/gguf/loader.go`**
   - Added Q6_K case to size calculation logic
   - 210 bytes per 256 elements

## Q6_K Block Format

Based on llama.cpp's implementation:

```
Block Size: 256 elements → 210 bytes (6.5625 bits/weight)

Structure:
- Ql[128]:     Low 4 bits of quantized values (2 values per byte)
- Qh[64]:      High 2 bits of quantized values (4 values per byte)
- Scales[16]:  int8 scale factors (one per 16 values)
- D:           float16 super-block scale

Total: 128 + 64 + 16 + 2 = 210 bytes
```

## Dequantization Algorithm

```go
for each block:
    d = convert float16 to float32

    for i = 0 to 255:
        // Extract 6-bit value (0-63)
        low = Ql[i/2] (nibble based on i%2)
        high = (Qh[i/4] >> (2*(i%4))) & 0x03
        q = low | (high << 4)  // Combine to 6 bits

        // Apply scale (symmetric quantization)
        scale_idx = i / 16  // One scale per 16 values
        value = d * Scales[scale_idx] * (q - 32)
```

Key features:
- **Symmetric quantization**: Center point at 32 (range: -32 to +31)
- **Hierarchical scaling**: Super-block scale (d) × per-group scale × quantized value
- **Efficient bit packing**: 6 bits split into 4+2 for cache-friendly access

## Test Results

All 15 tests pass:

```
✓ TestDequantizeQ6_K_ZeroBlock
✓ TestDequantizeQ6_K_KnownValues
✓ TestDequantizeQ6_K_Roundtrip
✓ TestDequantizeQ6_K_MultipleBlocks
✓ TestDequantizeQ6_K_SizeMismatch
✓ TestDequantizeQ6_KElement
✓ TestDequantizeQ6_KElement_OutOfBounds
✓ TestParseQ6_KBlock
✓ TestParseQ6_KBlock_TooSmall
✓ TestDequantizeQ6_KTensor
✓ TestDequantizeQ6_KTensor_WrongType
✓ TestQuantizeQ6_K
✓ TestQuantizeQ6_K_MultipleBlocks
✓ TestQ6_K_BitPacking
✓ TestQ6_K_SymmetricQuantization
```

## Performance Benchmarks

```
BenchmarkDequantizeQ6_K          317.27 MB/s    0 allocs/op
BenchmarkDequantizeQ6_KElement   20.77 ns/op    0 allocs/op
BenchmarkDequantizeQ6_KTensor    90514 ns/op    5 allocs/op
BenchmarkQuantizeQ6_K            111751 ns/op   1 alloc/op
```

Performance characteristics:
- **High throughput**: 317 MB/s dequantization speed
- **Low latency**: ~21 ns per element access
- **Zero allocations**: Core dequantization loop is allocation-free
- **Memory efficient**: 6.5625 bits per weight (vs 32 bits for float32)

## Integration

Q6_K is now fully integrated:

1. **GGUF Loading**: Models with Q6_K weights can be loaded via GGUF
2. **Memory Mapping**: Efficient mmap support for large tensors
3. **Lazy Dequantization**: On-demand element access via `At()`
4. **Eager Dequantization**: Full tensor conversion via `DequantizeQ6_KTensor()`
5. **MatMul Support**: Automatic dequantization in matrix operations

## Usage Example

```go
// Load Q6_K model from GGUF
ggufFile, _ := gguf.Parse("model_q6_k.gguf")
tensor, _ := ggufFile.LoadTensor("weight.tensor")

// Access single element (lazy dequantization)
value := tensor.At(0, 100)

// Convert entire tensor to float32 (eager)
floatTensor, _ := tensor.DequantizeQ6_KTensor(tensor)

// Use in matrix multiplication (auto-dequantizes)
result := tensor.MatMul(inputTensor, weightTensor)
```

## Comparison with Q5_K

| Feature | Q5_K | Q6_K |
|---------|------|------|
| Bits per weight | 5.5 | 6.5625 |
| Block size | 176 bytes | 210 bytes |
| Values per block | 256 | 256 |
| Quantization range | 0-31 | 0-63 |
| Precision | Lower | Higher |
| Model size | Smaller | Larger |
| Accuracy | Good | Better |

Q6_K offers better precision at the cost of ~19% more storage than Q5_K, while still achieving ~4.9x compression vs float32.

## Next Steps

The Q6_K implementation is production-ready and can be used for:
1. Loading and running Q6_K quantized models
2. End-to-end inference with Q6_K weights
3. Performance testing and optimization
4. Model quality evaluation vs Q5_K

All tests pass and the implementation follows the same patterns as Q5_K for consistency.
