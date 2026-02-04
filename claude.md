# Vibrant - Quick Reference for Claude

## What is Vibrant?

Vibrant is a **local, GPU-accelerated LLM code assistant** built in Go. It's a CLI tool that brings AI-powered coding assistance directly to your terminal, running entirely on your machine with no API keys or internet required.

## Key Features

- **GPU Accelerated**: Metal GPU support on Apple Silicon (6.4x speedup for large operations)
- **Pure Go Inference**: Custom GGUF model loader with no external dependencies
- **Agentic Behavior**: 15+ tools, multi-step planning, self-correction
- **Context-Aware**: RAG with semantic search (TF-IDF vector store)
- **Code Intelligence**: AST parsing, symbol extraction, cross-package references
- **Interactive Editing**: Diff-based file editing with automatic backups

## Current Status

✅ **Feature Complete** - Production-ready agentic code assistant!

**Just Completed**: Phase 11.1 - GPU Backend Foundation
- Metal GPU backend for Apple Silicon with 6.4x speedup
- Device abstraction layer (CPU/GPU)
- Automatic tensor migration
- Production-validated with comprehensive tests

**220+ tests passing** across 20 packages with strong coverage.

## Architecture

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/
│   ├── agent/        # Agentic behavior (planning, self-correction)
│   ├── assistant/    # Conversation & prompt handling
│   ├── codeintel/    # AST parsing & symbol extraction
│   ├── context/      # RAG, vector store, code indexing
│   ├── gpu/          # GPU abstraction & Metal backend
│   ├── inference/    # Pure Go inference engine
│   ├── tensor/       # Tensor operations (CPU & GPU)
│   ├── transformer/  # Transformer architecture
│   ├── tools/        # 15+ built-in tools
│   └── ...
├── test/             # Integration & benchmark tests
├── specs/            # Technical specifications
└── docs/             # Documentation
```

## Documentation Structure

**IMPORTANT**: Keep the repository root clean!

### Root Directory (only 2 markdown files allowed)
- `README.md` - User-facing documentation
- `PLAN.md` - Implementation roadmap and work plan

### Documentation Organization
- `specs/` - Technical specifications and API docs
- `docs/` - All other documentation:
  - `docs/phases/` - Phase summaries (PHASE*_SUMMARY.md)
  - `docs/plans/` - Strategic planning documents
  - `docs/results/` - Test results, profiling, benchmarks
  - `docs/implementation/` - Implementation notes
  - `docs/setup/` - Setup guides

**Rule**: Never create markdown files in the root except README.md and PLAN.md. All progress summaries, roadmaps, and notes go in `docs/`.

## Building & Testing

```bash
# Build (with GPU support on macOS)
make build

# Run tests
make test

# Run with coverage
make test-coverage

# Integration tests
go test ./test/integration/...

# Benchmarks
go test ./test/bench/... -bench=. -benchmem
```

## GPU Support

GPU acceleration via Metal is available on macOS with Apple Silicon:

```bash
# Use GPU
vibrant ask --device gpu "your question"

# Use CPU (default)
vibrant ask --device cpu "your question"

# Auto-detect
vibrant ask --device auto "your question"
```

**Performance**: GPU provides 6.4x speedup for large matrix operations (512×512+), but CPU is faster for single-token decode due to overhead.

## Key Packages

- **tensor** - Tensor operations with device abstraction (CPU/GPU)
- **gpu** - Metal GPU backend and device management
- **inference** - Pure Go GGUF model inference engine
- **agent** - Tool calling, planning, self-correction
- **codeintel** - AST parsing and symbol extraction
- **context** - RAG with TF-IDF vector store
- **tools** - Built-in tools (read_file, analyze_code, run_tests, etc.)

## Testing Philosophy

- Unit tests for all core functionality
- Integration tests for end-to-end workflows
- Benchmarks for performance-critical paths
- Mock engine for development (real inference optional)

## Common Tasks

### Adding a New Tool
1. Implement tool in `internal/tools/`
2. Register in `internal/tools/registry.go`
3. Add tests in `internal/tools/*_test.go`
4. Update docs if user-facing

### Adding a GPU Kernel
1. Implement kernel in `internal/gpu/metal/kernels/`
2. Add Metal shader in `internal/gpu/metal/shaders/`
3. Register in device backend
4. Add benchmarks and validation tests

### Working on Inference
1. Core logic in `internal/inference/`
2. Tensor ops in `internal/tensor/`
3. GPU kernels in `internal/gpu/`
4. Test with mock models in `test/fixtures/`

## LLM Inference Note

Vibrant uses a **mock engine by default** because real llama.cpp inference requires manual setup (git submodules). This is perfect for development and testing.

For real inference, we recommend:
- Using **Ollama** as the backend
- Or following `docs/llama-setup.md` for manual vendor setup

## Style & Conventions

- **Go**: Standard Go conventions, gofmt, golangci-lint
- **Testing**: Table-driven tests, descriptive test names
- **Comments**: Only when clarifying complex logic
- **Errors**: Wrap with context using `fmt.Errorf`
- **Logs**: Structured logging via standard log package

## Next Steps / Future Work

- Phase 11.2: KV-cache optimization with GPU
- Phase 11.3: Model-specific optimizations
- Phase 12: Plugin enhancements and distributed RAG
- Production polish: Error handling, edge cases, UX

## Quick Commands

```bash
# List models
vibrant model list

# Download model
vibrant model download qwen2.5-coder-7b-q5

# Ask question
vibrant ask "What is a goroutine?"

# Chat mode
vibrant chat

# With context
vibrant ask --context ./src "How does auth work?"
```

---

**Remember**: This project values clean code, comprehensive tests, and keeping the root directory organized. When in doubt, check the specs or ask!
