# Vibrant

A local, CPU and GPU-optimized LLM code assistant built in Go with advanced RAG and plugin capabilities.

## Overview

Vibrant is a command-line tool that brings AI-powered coding assistance directly to your terminal, running entirely on your local machine. Features both CPU and GPU (Metal on Apple Silicon) acceleration for fast inference. No internet connection or API keys required.

## Features

- 🚀 **GPU Accelerated**: Metal GPU support on Apple Silicon with 6.4x speedup for large operations
- 🖥️  **CPU-optimized**: Runs efficiently on CPU using quantized models (GGUF format)
- 🧠 **Context-aware**: Understands your codebase structure with semantic search (RAG)
- 🎯 **Auto-tuned**: Automatically selects the best model based on your system RAM
- 💬 **Interactive**: Rich terminal UI with syntax highlighting for 30+ languages
- 🔒 **Private**: All processing happens locally - your code never leaves your machine
- 🔌 **Extensible**: Plugin system for custom functionality
- 📝 **Git Integration**: Smart commit message generation with conventional commits
- ⌨️  **Tab-Completion**: Advanced shell completion for zsh, bash, and fish
- 🔍 **Semantic Search**: TF-IDF based vector store for code retrieval
- 🤖 **Agentic**: 15+ tools for multi-step workflows with self-correction
- 🧪 **Test & Build**: Integrated testing, building, and linting support
- ✏️ **Code Editing**: Diff-based file editing with automatic backups
- 🔬 **AST Analysis**: Deep code understanding with symbol extraction

## Status

✅ **Feature Complete** - Agentic code assistant with GPU acceleration!

**Current Phase**: Phase 11.1 - GPU Backend Foundation ✅ **COMPLETE**

**GPU Backend (Phase 11.1)** - Just completed!
- ✅ Metal GPU backend for Apple Silicon
- ✅ Device abstraction layer (CPU/GPU)
- ✅ GPU kernels for MatMul, Softmax, RMSNorm
- ✅ Tensor device migration (CPU ↔ GPU)
- ✅ Memory management with unified memory
- ✅ CLI integration with --device flag
- ✅ Performance: 6.4x speedup on 512×512 operations

**Performance Results**:
- Single-row (decode): CPU faster (low overhead)
- Medium ops (128×128): 1.37x GPU speedup
- Large ops (512×512): **6.4x GPU speedup**
- Validated: Zero error on critical decode path

**Recent Improvements** (Phase 10.10):
- ✅ Fixed critical Q5_K quantization bug (inference blocker)
- ✅ Fixed integration test infrastructure (auto-builds binary)
- ✅ Fixed inspect-gguf build failure
- ✅ Verified complete quantization support (Q4_K, Q5_K, Q6_K)
- ✅ All 200+ tests passing, no regressions

**Note on LLM Inference**: Currently uses a mock engine by default due to go-llama.cpp requiring manual setup (git submodules). This is perfect for development and testing. For real inference, we recommend using Ollama or manual vendor setup. See [docs/llama-setup.md](docs/llama-setup.md) for details.

### Completed Phases
- ✅ **Phase 1**: Project Setup & Foundation
- ✅ **Phase 2**: System Detection & Model Management
- ✅ **Phase 3**: LLM Integration (with mock engine)
- ✅ **Phase 4**: Code Context System
- ✅ **Phase 5**: Assistant Core Features
- ✅ **Phase 6**: CLI User Experience
- ✅ **Phase 7**: Advanced Features (RAG, Plugins, Git Integration)
- ✅ **Phase 8**: Testing & Optimization
- ✅ **Phase 9**: Agentic Behavior (Claude Code-inspired)
- ✅ **Phase 10**: Pure Go Inference Engine (no dependencies!)
- ✅ **Phase 11.1**: GPU Backend Foundation (Metal on Apple Silicon)

### GPU Acceleration (Phase 11.1)

**Metal GPU Backend for Apple Silicon**:
- Device abstraction layer supporting CPU and GPU
- GPU kernels: MatMul, Softmax, RMSNorm, element-wise ops
- Automatic CPU ↔ GPU tensor migration
- Unified memory optimization (~40 GB/s transfer speeds)
- Production-ready with comprehensive validation

**Performance Benchmarks**:
| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Decode (1×512) | 120 μs | 430 μs | CPU 3.6x faster* |
| Medium (128×128) | 435 μs | 318 μs | GPU 1.37x faster |
| Large (512×512) | 12.5 ms | 2.0 ms | **GPU 6.4x faster** |

*GPU overhead dominates for small operations; CPU used for decode step.

**CLI Integration**:
```bash
# Use GPU acceleration (Apple Silicon only)
vibrant ask --device gpu "your question"

# Use CPU (default, compatible everywhere)
vibrant ask --device cpu "your question"

# Auto-detect and use best device
vibrant ask --device auto "your question"
```

### Key Capabilities

**🤖 Agentic Behavior (Phase 9.1)**
- **Tool calling system** with 15+ built-in tools
- **Multi-step planning** with dependency tracking
- **Self-correction** with automatic retry strategies
- **Task decomposition** for complex workflows
- **Progress tracking** and result summarization

**🧠 Code Intelligence (Phase 9.2)**
- **AST parsing** for Go with symbol extraction
- **Symbol resolution** (functions, methods, types, structs, interfaces)
- **Dependency tracking** and import analysis
- **Cross-package references** and code navigation

**✏️ Interactive Editing (Phase 9.3)**
- **Diff-based editing** with unified diff format
- **Find/replace** operations across files
- **Automatic backups** before modifications
- **Patch application** with validation

**🧪 Testing & Building (Phase 9.4-9.5)**
- **Test execution** for Go, Python, Node.js
- **Auto-detect** test frameworks and build tools
- **Build integration** (go build, make, npm, pip)
- **Error parsing** and diagnostic reporting

**🔍 Quality Assurance (Phase 9.6)**
- **Linting integration** (golangci-lint, pylint, eslint)
- **Security scanning** support
- **Best practices** suggestions
- **Issue reporting** with context

**📊 RAG & Performance**
- **Semantic code search** using TF-IDF vector embeddings
- **Smart commit messages** with conventional commits
- **Plugin system** for extensibility (93.2% test coverage)
- **164+ unit tests** with comprehensive coverage
- **Performance**: <1µs for most operations

### Test Coverage
- `tokenizer`: 100%
- `plugin`: 93.2%
- `agent`: 89.1%
- `gguf`: 87.2%
- `codeintel`: 81.8%
- `system`: 82.1%
- `gpu`: 80.1% (new!)
- `diff`: 78.3%
- `tensor`: 78.2%
- `transformer`: 62.1%
- `assistant`: 59.6%
- `context`: 49.7%

**Total**: 220+ tests passing across 20 packages

## Installation

### Install from Source

```bash
# Clone and build
git clone https://github.com/xupit3r/vibrant.git
cd vibrant
make build

# Install to system path
make install
```

### Shell Completion (Recommended)

After installing, enable tab-completion for your shell:

```bash
# Zsh
vibrant completion zsh > ~/.zsh/completion/_vibrant
echo 'fpath=(~/.zsh/completion $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc

# Bash
vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant
source ~/.local/share/bash-completion/completions/vibrant

# Fish
vibrant completion fish > ~/.config/fish/completions/vibrant.fish
```

Restart your shell or source your config to activate completions.

## Usage

```bash
# List available models
vibrant model list

# Download a specific model (with tab-completion!)
vibrant model download <TAB>
vibrant model download qwen2.5-coder-7b-q5

# Show model information
vibrant model info qwen2.5-coder-3b-q4

# Ask a question (downloads model if needed)
vibrant ask "What is a goroutine?"

# Ask with GPU acceleration (Apple Silicon only)
vibrant ask --device gpu "Explain Go interfaces"

# With specific model (use tab to see available models)
vibrant ask --model <TAB>
vibrant ask --model qwen2.5-coder-7b-q5 "Explain Go interfaces"

# Interactive chat mode
vibrant chat

# Ask with context from specific files/directories
vibrant ask --context ./src "How does authentication work?"
```

### GPU Support

GPU acceleration is available on macOS with Apple Silicon (M-series chips):

```bash
# Automatic device selection (tries GPU, falls back to CPU)
vibrant ask --device auto "your question"

# Force GPU (fails if not available)
vibrant ask --device gpu "your question"

# Force CPU (default, works everywhere)
vibrant ask --device cpu "your question"
```

**When to use GPU**:
- ✅ Large batch operations (prefill phase)
- ✅ Matrix operations > 128×128
- ❌ Single-token decode (CPU is faster due to overhead)

GPU provides **6.4x speedup** for large matrix operations!

## Building from Source

**Requirements:**
- Go 1.21 or later
- macOS (for GPU support) or Linux/Windows (CPU only)
- C compiler for Metal integration (macOS only, optional)
- Make

### Quick Setup

```bash
# Clone repository
git clone https://github.com/xupit3r/vibrant.git
cd vibrant

# Check and install dependencies automatically
make install-deps

# Or just check what's missing
make check-deps

# Build (tries llama.cpp, falls back to mock if unavailable)
make build

# Run tests
make test
```

### Build Options

By default, `make build` **attempts to build with llama.cpp** for real LLM inference. If llama.cpp dependencies aren't available, it automatically falls back to a mock engine.

```bash
# Default: tries llama.cpp, falls back to mock
make build

# Force llama.cpp (fails if dependencies missing)
make build-llama

# Force mock engine (for development/testing)
make build-mock
```

See [docs/llama-setup.md](docs/llama-setup.md) for detailed setup instructions.

## Architecture

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/          # Private application code
│   ├── agent/        # Agentic behavior: planning, self-correction
│   ├── assistant/    # Conversation & prompt handling
│   ├── codeintel/    # AST parsing, symbol extraction
│   ├── context/      # Code indexing, RAG, vector store
│   ├── diff/         # Diff generation & git integration
│   ├── gpu/          # GPU device abstraction & Metal backend
│   ├── inference/    # Inference engine with GPU support
│   ├── model/        # Model management & caching
│   ├── llm/          # LLM inference engine
│   ├── plugin/       # Plugin system
│   ├── tensor/       # Tensor operations (CPU & GPU)
│   ├── tokenizer/    # Tokenization (BPE)
│   ├── tools/        # Tool registry (15+ tools)
│   ├── transformer/  # Transformer model architecture
│   ├── config/       # Configuration management
│   ├── system/       # System detection utilities
│   └── tui/          # Terminal UI components
├── test/             # Integration and benchmark tests
├── specs/            # Technical specifications
└── docs/             # Additional documentation
```

## Available Tools

Vibrant includes 15+ built-in tools for agentic workflows:

**File Operations**
- `read_file` - Read file contents
- `write_file` - Write content to files
- `list_directory` - List directory contents
- `backup_file` - Create file backups
- `replace_in_file` - Find and replace in files

**Code Analysis**
- `analyze_code` - AST-based code analysis
- `find_files` - Search files by pattern
- `grep` - Search patterns in files
- `get_file_info` - Get file metadata

**Editing & Diffs**
- `generate_diff` - Create unified diffs
- `apply_diff` - Apply patches to files

**Build & Test**
- `run_tests` - Execute tests (Go, Python, Node.js)
- `build` - Build projects (make, go, npm, pip)
- `lint` - Run linters (golangci-lint, pylint, eslint)

**Shell**
- `shell` - Execute shell commands with timeout

## Model Support

Vibrant currently supports the following models:

| Model | Parameters | RAM Required | Recommended For |
|-------|-----------|--------------|-----------------|
| Qwen 2.5 Coder 3B (Q4_K_M) | 3B | 4 GB | Systems with 6-10 GB RAM |
| Qwen 2.5 Coder 7B (Q4_K_M) | 7B | 8 GB | Systems with 10-14 GB RAM |
| Qwen 2.5 Coder 7B (Q5_K_M) | 7B | 10 GB | Systems with 10-16 GB RAM |
| Qwen 2.5 Coder 14B (Q5_K_M) | 14B | 18 GB | Systems with 16+ GB RAM |

Models are automatically downloaded from HuggingFace on first use.

## Performance

### GPU Benchmarks (Apple M1)

```
MatMul 1×512 (decode):   CPU:  120 μs  |  GPU:  430 μs  |  CPU 3.6x faster
MatMul 128×128:          CPU:  435 μs  |  GPU:  318 μs  |  GPU 1.37x faster
MatMul 512×512:          CPU: 12.5 ms  |  GPU:  2.0 ms  |  GPU 6.4x faster
```

GPU shines for large operations, CPU is better for small decode steps.

### CPU Benchmarks (Intel Core i5-1240P)

```
BenchmarkConversationAdd       20365    64.3 µs/op    201 KB/op
BenchmarkVectorStoreAdd      1254909     878 ns/op      1 KB/op
BenchmarkVectorStoreSearch     46449    25.4 µs/op     10 KB/op
BenchmarkDiffGenerate        1640283     727 ns/op      2 KB/op
BenchmarkSmartCommitMsg      1959770     617 ns/op    344 B/op
```

## Development

See [PLAN.md](PLAN.md) for the complete implementation plan and [specs/](specs/) for detailed technical specifications.

### Running Tests

```bash
# All tests
go test ./...

# With coverage
go test ./... -cover

# Integration tests
go test ./test/integration/...

# Benchmarks
go test ./test/bench/... -bench=. -benchmem
```

## License

TBD
