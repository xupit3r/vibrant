# Vibrant - GitHub Copilot Instructions

## Project Overview

Vibrant is a **local, GPU-accelerated LLM code assistant** written in Go. It's a CLI tool that runs language models locally on CPU or GPU (Metal on Apple Silicon) with no external dependencies or API keys required.

## Key Characteristics

- **Language**: Go 1.21+
- **GPU Backend**: Metal (macOS Apple Silicon)
- **Inference**: Pure Go GGUF model loader (no C dependencies)
- **Architecture**: Modular design with 20+ packages
- **Test Coverage**: 220+ tests with strong coverage across all packages
- **Status**: Feature complete, production-ready

## Code Style & Conventions

### Go Standards
- Follow standard Go conventions and idioms
- Use `gofmt` and `golangci-lint` for formatting/linting
- Prefer table-driven tests with descriptive names
- Keep functions focused and composable
- Use meaningful variable names (no abbreviations unless standard)

### Error Handling
- Always wrap errors with context: `fmt.Errorf("context: %w", err)`
- Return errors, don't panic (except in truly exceptional cases)
- Check all errors, no silent failures
- Use custom error types for specific error conditions

### Comments
- Only comment complex logic that needs clarification
- Don't state the obvious
- Use godoc format for public APIs
- Keep comments up-to-date with code changes

### Testing
- Write tests for all new functionality
- Aim for >80% coverage on new code
- Use `testify/assert` for assertions
- Mock external dependencies
- Name tests: `TestFunctionName_Scenario_ExpectedOutcome`

## Project Structure

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/          # Private packages (main codebase)
│   ├── agent/        # Agentic behavior, planning
│   ├── assistant/    # Conversation management
│   ├── codeintel/    # AST parsing, symbols
│   ├── context/      # RAG, vector store
│   ├── gpu/          # GPU abstraction, Metal backend
│   ├── inference/    # GGUF inference engine
│   ├── tensor/       # Tensor operations
│   ├── transformer/  # Transformer architecture
│   └── tools/        # Built-in tools (15+)
├── test/             # Integration tests, benchmarks
├── specs/            # Technical specifications
└── docs/             # Documentation
```

## Documentation Organization

**CRITICAL RULE**: Keep the repository root clean!

### Allowed in Root
- `README.md` - User documentation
- `PLAN.md` - Implementation roadmap
- Standard files (LICENSE, Makefile, go.mod, etc.)

### NOT Allowed in Root
- ❌ Phase summaries (use `docs/phases/`)
- ❌ Roadmaps (use `docs/plans/`)
- ❌ Test results (use `docs/results/`)
- ❌ Implementation notes (use `docs/implementation/`)
- ❌ Any other markdown files

### Documentation Structure
```
docs/
├── phases/          # Phase completion summaries
├── plans/           # Strategic planning documents
├── results/         # Test results, benchmarks, profiling
├── implementation/  # Technical implementation notes
└── setup/          # User setup guides
```

**When creating documentation**: Always place it in the appropriate `docs/` subdirectory, never in the root.

## Common Tasks

### Adding a New Feature
1. Check `PLAN.md` for roadmap alignment
2. Create or update relevant spec in `specs/`
3. Implement in appropriate `internal/` package
4. Write comprehensive tests
5. Update `README.md` if user-facing
6. Run `make test` and `make lint` before committing

### Working with GPU Code
- GPU code is in `internal/gpu/metal/`
- Tensor operations support both CPU and GPU
- Use device abstraction layer (`tensor.Device`)
- Always benchmark performance changes
- Validate correctness with unit tests

### Adding a Tool
1. Implement in `internal/tools/`
2. Register in `internal/tools/registry.go`
3. Add tests in `internal/tools/*_test.go`
4. Document in tool's godoc comment

### Modifying Inference Engine
- Core logic: `internal/inference/`
- Tensor ops: `internal/tensor/`
- Quantization: `internal/gguf/quant/`
- Always validate with test models in `test/fixtures/`

## Build & Test Commands

```bash
# Build
make build

# Run tests
make test

# Run with coverage
make test-coverage

# Run linters
make lint

# Integration tests
go test ./test/integration/...

# Benchmarks
go test ./test/bench/... -bench=. -benchmem
```

## Important Notes

### LLM Inference
- **Mock engine** is used by default (no dependencies)
- Real inference requires llama.cpp vendor setup
- See `docs/setup/llama-setup.md` for details
- Mock is perfect for development/testing

### GPU Acceleration
- Only available on macOS with Apple Silicon
- Provides 6.4x speedup for large operations (512×512+)
- CPU is faster for small operations due to overhead
- Device selection: `--device cpu|gpu|auto`

### Performance
- CPU is optimized with SIMD operations
- GPU uses Metal with unified memory
- Benchmarks are in `test/bench/`
- Profile before optimizing: `go test -cpuprofile`

## Code Patterns

### Tensor Operations
```go
// Always use device abstraction
device := tensor.NewCPUDevice()
t := tensor.NewWithDevice(data, shape, device)

// Migrate between devices
t.MigrateToDevice(gpuDevice)
```

### Error Wrapping
```go
if err != nil {
    return fmt.Errorf("loading model: %w", err)
}
```

### Tool Implementation
```go
type MyTool struct {
    name        string
    description string
}

func (t *MyTool) Name() string { return t.name }
func (t *MyTool) Description() string { return t.description }
func (t *MyTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    // Implementation
}
```

## Testing Patterns

### Table-Driven Tests
```go
func TestMyFunction(t *testing.T) {
    tests := []struct {
        name     string
        input    int
        expected int
    }{
        {"positive", 5, 10},
        {"negative", -5, -10},
        {"zero", 0, 0},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := MyFunction(tt.input)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

## Git Workflow

- Commit messages: Use conventional commits format
- Keep commits focused and atomic
- Run tests before committing
- Update docs when changing APIs

## Performance Considerations

- Prefer CPU for small operations (<128×128)
- Use GPU for large batch operations
- Pre-allocate tensors when possible
- Reuse buffers in hot paths
- Profile before optimizing

## Common Pitfalls to Avoid

1. **Don't create markdown files in root** - Use `docs/` subdirectories
2. **Don't skip error checks** - Always handle errors explicitly
3. **Don't assume GPU is available** - Always check device type
4. **Don't modify tensors after migration** - Clone if needed
5. **Don't use panic** - Return errors instead

## Resources

- `PLAN.md` - Full implementation roadmap
- `README.md` - User documentation
- `specs/` - Technical specifications
- `docs/setup/` - Setup guides
- `test/` - Examples of usage patterns

## Questions?

Check the following in order:
1. `README.md` for user-facing info
2. `PLAN.md` for implementation details
3. `specs/` for technical specifications
4. Relevant test files for usage examples
5. Code comments for specific logic

---

**Remember**: Clean root directory, comprehensive tests, clear error messages, and follow Go idioms!
