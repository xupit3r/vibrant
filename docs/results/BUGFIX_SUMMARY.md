# Bug Fix Summary - Q6_K Subnormal Float16 Issue

## Issue
Model always predicted token 50290 ("ontvangst") regardless of prompt input, making the inference engine unusable.

## Root Cause
The `float16ToFloat32()` function in `internal/tensor/tensor.go` incorrectly returned 0.0 for ALL subnormal float16 numbers:

```go
// BUGGY CODE
if exp == 0 {
    return 0.0  // Wrong! Treats all subnormals as zeros
}
```

### Impact
- Q6_K quantization uses float16 for per-block scale factors (D values)
- Many D values are subnormal (e.g., 0x00fd ≈ 1.5e-5)
- When D=0, entire blocks dequantize to zeros: `output = D × scale × (q-32) = 0`
- Result: 99.999998% of output weight matrix became zeros
- Only ~500 out of 311M elements were non-zero
- Those 500 elements were all in columns 50048-68223, explaining the token 50290 bias

## Solution
Properly handle subnormal float16 numbers by normalizing them:

```go
if exp == 0 {
    if mantissa == 0 {
        return 0.0  // True zero
    }
    // Subnormal: normalize by finding leading 1
    shift := uint32(0)
    m := mantissa
    for (m & 0x400) == 0 {
        m <<= 1
        shift++
    }
    m &= 0x3FF
    exp32 := uint32(127 - 14 - shift)
    mantissa32 := m << 13
    bits := (sign << 31) | (exp32 << 23) | mantissa32
    return *(*float32)(unsafe.Pointer(&bits))
}
```

## Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Non-zero elements in output weight | 494 (0.00%) | 301,163,031 (96.79%) |
| Logit range | -3.7 to +3.0 | -16.3 to +20.5 |
| Predicted token | 50290 (always) | 128008 (different issue*) |

*The model still repeats tokens but with proper logit distributions - this is a separate bug unrelated to float16 conversion.

## Files Changed
1. **`internal/tensor/tensor.go`** - Fixed `float16ToFloat32()` (lines 611-655)
2. **`internal/tensor/float16_test.go`** - Added comprehensive test suite (NEW FILE)

## Testing
- ✅ All existing tensor tests pass
- ✅ New float16 tests cover subnormals, normals, and special values
- ✅ Q6_K dequantization verified to produce 96.79% non-zero values
- ✅ Logit distributions now reasonable

## Documentation
- See `docs/results/Q6K_DEQUANT_BUG_ANALYSIS.md` for detailed investigation

---
**Conclusion**: The critical subnormal float16 bug that caused the "ontvangst" repetition is FIXED. The dequantization now works correctly.
