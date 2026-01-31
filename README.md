# Vibrant

A local, CPU-optimized LLM code assistant built in Go with advanced RAG and plugin capabilities.

## Overview

Vibrant is a command-line tool that brings AI-powered coding assistance directly to your terminal, running entirely on your local machine using CPU inference. No internet connection or API keys required.

## Features

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

✅ **Feature Complete** - Agentic code assistant with Claude Code-inspired capabilities!

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
- `agent`: 100%
- `codeintel`: 100%
- `tools`: 100%
- `assistant`: 59.6%
- `context`: 49.7%
- `diff`: 78.3%
- `plugin`: 93.2%
- `system`: 82.7%

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

# With specific model (use tab to see available models)
vibrant ask --model <TAB>
vibrant ask --model qwen2.5-coder-7b-q5 "Explain Go interfaces"

# Interactive chat mode
vibrant chat

# Ask with context from specific files/directories
vibrant ask --context ./src "How does authentication work?"
```

## Building from Source

**Requirements:**
- Go 1.21 or later
- C++ compiler (gcc/clang) for llama.cpp integration (optional - falls back to mock)
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
│   ├── model/        # Model management & caching
│   ├── llm/          # LLM inference engine
│   ├── plugin/       # Plugin system
│   ├── tools/        # Tool registry (15+ tools)
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

Benchmark results on 12th Gen Intel Core i5-1240P:

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
