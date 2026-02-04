# Vibrant - AI-Assisted Development Guide

Vibrant is a local CPU-optimized LLM code assistant built in Go with agentic capabilities, RAG, and plugin system.

## Build, Test, and Lint

### Building
```bash
# Default: Pure Go custom engine (no CGO required)
make build

# With llama.cpp (requires C++ compiler and CGO)
make build-llama

# Mock engine (for testing without dependencies)
make build-mock
```

### Testing
```bash
# Run all tests
go test ./...
make test

# With coverage
make test-coverage

# Single package
go test ./internal/agent -v

# Single test
go test ./internal/agent -run TestNewAgent -v

# Benchmarks
make bench

# Specific benchmarks
make bench-tensor
make bench-inference
```

### Linting
```bash
make lint
# Or directly: golangci-lint run
```

## Architecture Overview

Vibrant has a layered architecture with clear separation of concerns:

### Core Layers
- **cmd/vibrant**: CLI entry point (cobra commands)
- **internal/agent**: Agentic behavior - planning, execution, self-correction
- **internal/assistant**: Conversation & prompt handling
- **internal/tools**: Tool registry (15+ tools for file ops, code analysis, build/test)
- **internal/llm**: LLM inference engines (custom, llama.cpp, mock)
- **internal/context**: Code indexing, RAG, TF-IDF vector store
- **internal/codeintel**: AST parsing, symbol extraction (Go focused)
- **internal/model**: Model management, downloading, caching
- **internal/gguf**: GGUF file format parsing
- **internal/tensor**: Tensor operations for custom inference
- **internal/transformer**: Transformer architecture implementation
- **internal/inference**: High-level inference orchestration

### Build Tags and Engine Selection
The project uses build tags to switch between inference engines:
- **Default (no tag)**: Custom pure Go engine (CGO_ENABLED=0)
- **`-tags llama`**: llama.cpp via CGO (CGO_ENABLED=1)
- The `internal/llm` package abstracts the engine choice

### Tool System Architecture
- Tools implement the `Tool` interface in `internal/tools/registry.go`
- Each tool has: Name, Description, Parameters (schema), Execute method
- Agent orchestrates tool calls through the registry
- Tools are organized by category: file_tools, code_tools, edit_tools, build_tools, shell_tools

## Key Conventions

### Testing Patterns
- Test files follow `*_test.go` convention
- Use table-driven tests where appropriate
- 35 test files cover critical paths
- Mock engine available for testing without real inference
- Use `t.Run()` for subtests with descriptive names

### Code Organization
- Keep public APIs in package-level files (e.g., `agent.go`, `registry.go`)
- Internal/private helpers in separate files or same file as private functions
- Types usually defined near their primary usage or in dedicated `types.go`
- Build tag constraints at top of file: `// +build llama`

### Error Handling
- Wrap errors with context: `fmt.Errorf("description: %w", err)`
- Return errors rather than panicking in library code
- Use `context.Context` for cancellation in long-running operations

### Naming Conventions
- Interface names often match their implementation without "I" prefix (e.g., `Tool` interface)
- Getters don't use "Get" prefix (e.g., `Name()` not `GetName()`)
- Private fields use camelCase, exported use PascalCase
- Test functions: `TestFunctionName` or `TestFunctionName_Scenario`

### Documentation Requirements
- All technical specs go in `specs/` directory
- Phase summaries and plans go in `docs/` subdirectories (phases/, plans/, results/, implementation/, setup/)
- Keep root clean: only `README.md` and `PLAN.md` in root
- Update specs when implementation changes

### Model and LLM Integration
- Models are GGUF format from HuggingFace
- Model metadata in registry defines RAM requirements
- Model files cached in user's home directory
- Custom inference engine is pure Go, no CGO required
- llama.cpp engine requires manual vendor setup (see docs/llama-setup.md)

### Context and RAG System
- File indexer respects `.gitignore` patterns
- TF-IDF based vector store for semantic search
- Files indexed with metadata (language, size, test/generated flags)
- Context chunking for large files

### Agent Behavior
- Max steps default: 10 (configurable via `SetMaxSteps`)
- Actions have Tool, Parameters, and Reasoning fields
- Plans track goal, actions, and status
- Context cancellation supported throughout

### Dependencies Management
- Use `go mod tidy` to clean up dependencies
- Scripts in `scripts/` for dependency checking and llama.cpp setup
- Don't commit vendor/ directory

## Common Patterns

### Adding a New Tool
1. Create tool struct in appropriate `internal/tools/*_tools.go` file
2. Implement the `Tool` interface (Name, Description, Parameters, Execute)
3. Register tool in the default registry
4. Add tests in corresponding `*_test.go` file

### Working with Build Tags
- Custom engine: No special tags needed
- llama.cpp: Use `// +build llama` at top of file
- Test with: `go test -tags llama ./...`

### Handling Model Files
- Model paths resolved via `internal/model/manager.go`
- Download logic in downloader, with progress tracking
- Model selection based on system RAM detection

### AST Analysis (Go Code)
- Use `internal/codeintel` for parsing Go source
- Extracts: functions, methods, types, structs, interfaces
- Supports dependency tracking and import analysis

## Performance Considerations

- Most operations < 1µs (vector store, diff generation)
- Use streaming for LLM inference to show progress
- Context window management critical for memory
- Tensor operations optimized for CPU

## Shell Completion

After installation, users should set up completion:
```bash
# See docs/setup/COMPLETION-QUICKSTART.md for details
vibrant completion zsh > ~/.zsh/completion/_vibrant
vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant
vibrant completion fish > ~/.config/fish/completions/vibrant.fish
```
