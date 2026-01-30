# Vibrant

A local, CPU-optimized LLM code assistant built in Go.

## Overview

Vibrant is a command-line tool that brings AI-powered coding assistance directly to your terminal, running entirely on your local machine using CPU inference. No internet connection or API keys required.

## Features

- 🖥️  **CPU-optimized**: Runs efficiently on CPU using quantized models (GGUF format)
- 🧠 **Context-aware**: Understands your codebase structure and provides relevant assistance
- 🎯 **Auto-tuned**: Automatically selects the best model based on your system RAM
- 💬 **Interactive**: Rich terminal UI for seamless conversation
- 🔒 **Private**: All processing happens locally - your code never leaves your machine

## Status

🚧 **Under Development** - See [PLAN.md](PLAN.md) for implementation roadmap.

### Completed Phases
- ✅ **Phase 1**: Project Setup & Foundation
- ✅ **Phase 2**: System Detection & Model Management
- ✅ **Phase 3**: LLM Integration (with mock engine)

### Current Status
Core infrastructure complete with:
- RAM detection and model selection
- Model download and caching system
- Model registry with Qwen 2.5 Coder series
- LLM inference interface (mock implementation)
- CLI commands: `model list/info/download/remove`, `ask`

### Next Up
- Phase 4: Code Context System
- Phase 5: Assistant Core Features
- Phase 6: CLI User Experience enhancements

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
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/xupit3r/vibrant.git
cd vibrant

# Build (standard - uses mock engine)
go build -o vibrant ./cmd/vibrant

# Build with llama.cpp (requires C++ compiler)
CGO_ENABLED=1 go build -tags llama -o vibrant ./cmd/vibrant
```

## Architecture

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/          # Private application code
│   ├── model/        # Model management & caching
│   ├── llm/          # LLM inference engine
│   ├── context/      # Code indexing & retrieval
│   ├── assistant/    # Conversation & prompt handling
│   ├── config/       # Configuration management
│   └── system/       # System detection utilities
├── pkg/              # Public libraries (if any)
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

## Development

See [PLAN.md](PLAN.md) for the complete implementation plan and [specs/](specs/) for detailed technical specifications.

## License

TBD
