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
- 🔍 **Semantic Search**: TF-IDF based vector store for code retrieval

## Status

✅ **Core Complete** - Advanced features implemented!

### Completed Phases
- ✅ **Phase 1**: Project Setup & Foundation
- ✅ **Phase 2**: System Detection & Model Management
- ✅ **Phase 3**: LLM Integration (with mock engine)
- ✅ **Phase 4**: Code Context System
- ✅ **Phase 5**: Assistant Core Features
- ✅ **Phase 6**: CLI User Experience
- ✅ **Phase 7**: Advanced Features (RAG, Plugins, Git Integration)
- ✅ **Phase 8**: Testing & Optimization

### Key Capabilities
- **Multi-turn conversations** with intelligent context pruning
- **Semantic code search** using vector embeddings (TF-IDF)
- **Diff generation** and smart commit messages
- **Plugin system** for extensibility (93.2% test coverage)
- **85+ unit tests** with comprehensive integration tests
- **Performance benchmarks**: <1µs for most operations

### Test Coverage
- `assistant`: 59.6%
- `context`: 49.7%
- `diff`: 78.3%
- `plugin`: 93.2%
- `system`: 82.7%

## Usage

```bash
# List available models
vibrant model list

# Download a specific model
vibrant model download qwen2.5-coder-7b-q5

# Show model information
vibrant model info qwen2.5-coder-3b-q4

# Ask a question (downloads model if needed)
vibrant ask "What is a goroutine?"

# With specific model
vibrant ask --model qwen2.5-coder-7b-q5 "Explain Go interfaces"

# Interactive chat mode
vibrant chat

# Ask with context from specific files/directories
vibrant ask --context ./src "How does authentication work?"
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/xupit3r/vibrant.git
cd vibrant

# Run tests
make test

# Run benchmarks
make bench

# Build (standard - uses mock engine)
make build

# Build with llama.cpp (requires C++ compiler)
CGO_ENABLED=1 go build -tags llama -o vibrant ./cmd/vibrant
```

## Architecture

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/          # Private application code
│   ├── assistant/    # Conversation & prompt handling
│   ├── context/      # Code indexing, RAG, vector store
│   ├── diff/         # Diff generation & git integration
│   ├── model/        # Model management & caching
│   ├── llm/          # LLM inference engine
│   ├── plugin/       # Plugin system
│   ├── config/       # Configuration management
│   ├── system/       # System detection utilities
│   └── tui/          # Terminal UI components
├── test/             # Integration and benchmark tests
├── specs/            # Technical specifications
└── docs/             # Additional documentation
```

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
