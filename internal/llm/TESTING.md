# LLM Package Testing Guide

This document explains how to test the LLM package, including the CustomEngine integration with real GGUF models.

## Quick Start

### Running Unit Tests

Unit tests that don't require real models can be run with:

```bash
go test ./internal/llm -v
```

These tests verify:
- Engine interface implementation
- Configuration conversion logic
- Error handling for invalid paths
- Basic functionality without model files

### Running Integration Tests with Real Models

Integration tests require a GGUF model file. To run them:

1. **Download a small test model** (recommended: TinyLlama or similar):
   ```bash
   # Example: Download TinyLlama 1.1B (GGUF format)
   # Place it in testdata/ directory or any location
   ```

2. **Set the test model path** via environment variable:
   ```bash
   export VIBRANT_TEST_MODEL=/path/to/your/model.gguf
   ```

3. **Update the test helper** in `engine_test.go`:
   ```go
   func getTestModelPath() string {
       // Check environment variable
       if path := os.Getenv("VIBRANT_TEST_MODEL"); path != "" {
           return path
       }
       // Or hardcode for local testing
       return "/path/to/test/model.gguf"
   }
   ```

4. **Run integration tests**:
   ```bash
   go test ./internal/llm -v -run TestCustomEngine
   ```

## Test Coverage

### What's Tested

#### Without Real Models (Always Runs)
- ✅ CustomEngine interface implementation
- ✅ Invalid path error handling
- ✅ Configuration conversion (LoadOptions → inference.Config)
- ✅ Multiple configuration scenarios

#### With Real Models (Requires GGUF file)
- ✅ Engine creation and initialization
- ✅ Text generation (blocking)
- ✅ Text generation (streaming)
- ✅ Token counting
- ✅ Context cancellation handling
- ✅ Resource cleanup (Close)
- ✅ String representation
- ✅ NewCustomEngineWithConfig

### Current Coverage

When running without models (default):
```
coverage: ~27% of statements
```

When running with real models:
```
coverage: ~85%+ of statements (estimated)
```

The low coverage without models is expected since most of the actual functionality requires a loaded model.

## Test Structure

### engine_test.go

Contains comprehensive tests for both the old LlamaEngine (CGO-based) and new CustomEngine (pure Go).

**CustomEngine Tests:**

1. **TestCustomEngineCreation**: Basic engine instantiation
2. **TestCustomEngineInvalidPath**: Error handling for missing files
3. **TestCustomEngineConfigConversion**: LoadOptions conversion verification
4. **TestCustomEngineGenerate**: Blocking text generation
5. **TestCustomEngineGenerateStream**: Streaming text generation
6. **TestCustomEngineTokenCount**: Tokenization accuracy
7. **TestCustomEngineClose**: Resource cleanup
8. **TestCustomEngineWithConfig**: Advanced configuration
9. **TestCustomEngineString**: String representation
10. **TestCustomEngineContextCancellation**: Context handling

### Helper Functions

- `getTestModelPath()`: Returns path to test model (or empty to skip)
- `truncateStr()`: Safely truncates long strings for logging

## Creating Test Models

For testing without downloading large models, you can:

1. **Use a tiny model**: TinyLlama-1.1B-Chat (GGUF) is ~600MB
2. **Create a minimal GGUF**: Use `gguf-py` to create a minimal test fixture
3. **Mock the GGUF loader**: For unit tests, mock the entire inference stack

### Example: Minimal Test Model

To create a minimal GGUF file for testing (Python required):

```python
# create_test_model.py
import gguf
import numpy as np

writer = gguf.GGUFWriter("testdata/tiny_test.gguf", "test")

# Add minimal metadata
writer.add_architecture("llama")
writer.add_context_length(512)
writer.add_embedding_length(128)
writer.add_head_count(4)
writer.add_layer_count(2)

# Add minimal weights (random for testing)
writer.add_tensor("token_embd.weight", np.random.randn(100, 128).astype(np.float32))
# ... add other required tensors

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

## CI/CD Integration

For continuous integration:

1. **Skip integration tests by default**:
   ```bash
   go test ./internal/llm -short
   ```

2. **Run full tests in dedicated jobs**:
   ```yaml
   # .github/workflows/test.yml
   - name: Download test model
     run: |
       wget https://example.com/tiny-model.gguf -O /tmp/test.gguf
       export VIBRANT_TEST_MODEL=/tmp/test.gguf

   - name: Run integration tests
     run: go test ./internal/llm -v
   ```

3. **Use model caching**:
   ```yaml
   - name: Cache test model
     uses: actions/cache@v3
     with:
       path: ~/.cache/vibrant-test-models
       key: test-models-v1
   ```

## Performance Testing

### Benchmarks

Run benchmarks to verify performance:

```bash
go test ./internal/llm -bench=. -benchmem
```

### Memory Profiling

Profile memory usage with a real model:

```bash
go test ./internal/llm -run TestCustomEngineGenerate -memprofile=mem.prof
go tool pprof mem.prof
```

### CPU Profiling

Profile CPU usage during inference:

```bash
go test ./internal/llm -run TestCustomEngineGenerate -cpuprofile=cpu.prof
go tool pprof cpu.prof
```

## Troubleshooting

### Tests Skip Automatically

**Problem**: All CustomEngine tests show "SKIP"

**Solution**: Set `VIBRANT_TEST_MODEL` environment variable or update `getTestModelPath()` to return a valid path.

### Out of Memory During Tests

**Problem**: Tests crash with OOM errors

**Solution**: Use a smaller model or increase available memory. TinyLlama-1.1B requires ~1-2GB RAM.

### Slow Test Execution

**Problem**: Tests take too long to run

**Solution**:
- Reduce MaxTokens in test configurations
- Use a smaller model
- Run specific tests instead of full suite

### GGUF Parse Errors

**Problem**: "failed to open GGUF file" errors

**Solution**:
- Verify model file is valid GGUF format (not GGML or other)
- Check file permissions
- Ensure model architecture is supported (currently: llama)

## Best Practices

1. **Use small models for testing**: Don't use 70B models for unit tests
2. **Test with multiple model sizes**: Verify scalability
3. **Test error paths**: Ensure graceful degradation
4. **Monitor resource usage**: Check for memory leaks
5. **Test context cancellation**: Verify cleanup on interruption
6. **Validate outputs**: Don't just check non-empty, verify quality

## Future Improvements

- [ ] Automated test model downloading
- [ ] Mocked GGUF loader for pure unit tests
- [ ] Performance regression tests
- [ ] Multi-threaded generation tests
- [ ] Stress tests with long contexts
- [ ] Comparison tests (CustomEngine vs LlamaEngine)
