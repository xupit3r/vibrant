# CLI Interface Specification

## Overview
The CLI interface provides both single-shot and interactive modes for interacting with the Vibrant code assistant.

## Command Structure

```
vibrant
├── ask [question]           # Single-shot query
├── chat                     # Interactive mode
├── model                    # Model management
│   ├── list                # List available models
│   ├── download <id>       # Download a model
│   ├── remove <id>         # Remove a model
│   ├── info <id>           # Show model details
│   └── cache               # Cache management
├── config                   # Configuration management
│   ├── show                # Display config
│   ├── get <key>           # Get config value
│   ├── set <key> <value>   # Set config value
│   └── reset               # Reset to defaults
└── version                  # Version information
```

## Status
- **Current**: Partial implementation (basic commands)
- **Last Updated**: 2026-01-30
- **Implementation**: Phase 1 (in progress)
