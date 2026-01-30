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

## Planned Usage

```bash
# Interactive mode
vibrant chat

# Single query
vibrant ask "how do I implement a binary search in Go?"

# With project context
vibrant ask --context ./src "explain this architecture"
```

## Architecture

```
vibrant/
├── cmd/vibrant/       # CLI entry point
├── internal/          # Private application code
│   ├── model/        # Model management & inference
│   ├── context/      # Code indexing & retrieval
│   ├── assistant/    # Conversation & prompt handling
│   ├── config/       # Configuration management
│   └── system/       # System detection utilities
├── pkg/              # Public libraries (if any)
├── specs/            # Technical specifications
└── docs/             # Additional documentation
```

## Development

See [PLAN.md](PLAN.md) for the complete implementation plan and specifications.

## License

TBD
