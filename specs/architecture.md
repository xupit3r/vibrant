# Architecture Specification

## Overview
Vibrant is a local LLM-powered code assistant with a modular architecture designed for CPU-efficient inference and context-aware assistance.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Layer (cmd/)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ chat command │  │ ask command  │  │ config cmd   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│              Application Core (internal/)                    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Assistant Orchestrator                      │  │
│  │  - Handles user queries                                │  │
│  │  - Coordinates between components                      │  │
│  │  - Manages conversation flow                           │  │
│  └────┬──────────────┬──────────────┬──────────────────┘  │
│       │              │              │                       │
│  ┌────▼──────┐  ┌───▼────────┐  ┌──▼───────────────────┐  │
│  │  Context  │  │   Model    │  │   Config Manager     │  │
│  │  System   │  │  Manager   │  │  - Settings          │  │
│  │  - Index  │  │  - Load    │  │  - Preferences       │  │
│  │  - Search │  │  - Infer   │  │  - Validation        │  │
│  └───────────┘  └────────────┘  └──────────────────────┘  │
│                       │                                      │
│                  ┌────▼────────┐                            │
│                  │   System    │                            │
│                  │  Detection  │                            │
│                  │  - RAM      │                            │
│                  │  - CPU info │                            │
│                  └─────────────┘                            │
└──────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                    External Resources                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ llama.cpp    │  │ Model Files  │  │  User Code   │      │
│  │ (via CGO)    │  │ (.gguf)      │  │  Repository  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
vibrant/
├── cmd/
│   └── vibrant/
│       ├── main.go              # Application entry point
│       ├── root.go              # Root cobra command
│       ├── chat.go              # Interactive chat command
│       ├── ask.go               # Single-shot query command
│       └── config.go            # Configuration management commands
│
├── internal/
│   ├── assistant/
│   │   ├── assistant.go         # Main orchestrator
│   │   ├── conversation.go      # Conversation state management
│   │   ├── prompts.go           # Prompt templates
│   │   └── templates/           # Prompt template files
│   │
│   ├── model/
│   │   ├── manager.go           # Model lifecycle management
│   │   ├── inference.go         # Inference wrapper
│   │   ├── registry.go          # Model metadata registry
│   │   ├── downloader.go        # Download from HuggingFace
│   │   ├── selector.go          # RAM-based model selection
│   │   └── cache.go             # Model file caching
│   │
│   ├── context/
│   │   ├── indexer.go           # File tree indexing
│   │   ├── gitignore.go         # .gitignore parsing
│   │   ├── retriever.go         # Context retrieval
│   │   ├── chunker.go           # Code chunking
│   │   └── summarizer.go        # Project summarization
│   │
│   ├── config/
│   │   ├── config.go            # Configuration struct
│   │   ├── loader.go            # Load from files/env
│   │   ├── defaults.go          # Default values
│   │   └── validator.go         # Configuration validation
│   │
│   └── system/
│       ├── detect.go            # System detection utilities
│       ├── ram.go               # RAM detection
│       └── platform.go          # Platform-specific code
│
├── pkg/                         # Public libraries (future)
│
├── specs/                       # Technical specifications
│   ├── architecture.md          # This file
│   ├── model-management.md
│   ├── llm-integration.md
│   ├── context-system.md
│   ├── assistant-core.md
│   ├── cli-interface.md
│   ├── configuration.md
│   └── dependencies.md
│
├── docs/                        # Additional documentation
├── .gitignore
├── go.mod
├── go.sum
├── README.md
└── PLAN.md
```

## Component Interactions

### Query Flow (ask command)

```
User Input
    │
    ▼
┌───────────────┐
│ CLI Parser    │ Parse flags and arguments
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Config Load   │ Load user preferences
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Model Manager │ Select/load appropriate model
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Context System│ Gather relevant code context
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Assistant     │ Build prompt with context
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM Inference │ Generate response (streaming)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Output Format │ Format and display to user
└───────────────┘
```

### Chat Mode Flow

```
User starts chat
    │
    ▼
Initialize ─────► Load Model ─────► Load Context (initial)
    │                                       │
    ▼                                       │
┌─────────────────────────────────────────┴──┐
│          Conversation Loop                  │
│                                             │
│  User Input → Build Prompt → Inference     │
│       ▲            │              │         │
│       │            ▼              ▼         │
│       └───── Update History ◄─ Response    │
│                                             │
└─────────────────────────────────────────────┘
    │
    ▼
Save session (optional) → Exit
```

## Data Flow

### Model Loading
```
User starts Vibrant
    │
    ▼
System Detection (RAM, CPU)
    │
    ▼
Model Registry Lookup
    │
    ▼
Check Cache (~/.vibrant/models/)
    │
    ├─ Found ──────────────────┐
    │                          │
    └─ Not Found              │
        │                      │
        ▼                      │
   Download from              │
   HuggingFace                │
        │                      │
        └──────────────────────┤
                               │
                               ▼
                        Load into Memory
                               │
                               ▼
                        Ready for Inference
```

### Context Building
```
User Query + Project Path
    │
    ▼
Walk Directory Tree
    │
    ├─ Skip: node_modules, .git, etc.
    │
    ▼
Build File Index
    │
    ▼
Determine Relevant Files
    │  (based on query keywords,
    │   file types, recency)
    │
    ▼
Extract Code Chunks
    │
    ▼
Estimate Token Count
    │
    ▼
Prioritize & Trim to fit context window
    │
    ▼
Format Context for Prompt
```

## State Management

### Model State
- **Location**: In-memory (internal/model)
- **Lifecycle**: Loaded once, persists during session
- **Thread Safety**: Mutex-protected for concurrent access

### Conversation State
- **Location**: In-memory (internal/assistant)
- **Persistence**: Optional save to ~/.vibrant/sessions/
- **Format**: JSON with timestamp, messages, metadata

### Configuration State
- **Location**: ~/.vibrant/config.yaml
- **Reload**: On-demand or SIGHUP
- **Validation**: At load time with sensible defaults

### Context Cache
- **Location**: In-memory LRU cache
- **TTL**: Until project files change (inotify/fsnotify)
- **Size Limit**: Configurable, default 100MB

## Concurrency Model

### Goroutine Usage
1. **Model Inference**: Synchronous (llama.cpp handles internal threading)
2. **File Indexing**: Concurrent (goroutine pool, ~NumCPU workers)
3. **Streaming Output**: Buffered channel for token streaming
4. **Downloads**: Single goroutine with progress updates

### Synchronization
- Model access: `sync.Mutex` or `sync.RWMutex`
- File operations: Channel-based coordination
- Configuration: `sync.RWMutex` for concurrent reads

## Error Handling Strategy

### Error Types
1. **Fatal**: Model load failure, invalid config → Exit with error
2. **Recoverable**: Network issues, context errors → Retry or degrade
3. **User**: Invalid input, missing files → Display help and continue

### Logging Levels
- **DEBUG**: Detailed inference stats, context building steps
- **INFO**: Model loading, command execution, normal operations
- **WARN**: Degraded functionality, using fallbacks
- **ERROR**: Failures that impact functionality but don't crash
- **FATAL**: Unrecoverable errors

## Security Considerations

1. **Model Verification**: SHA256 checksums for downloaded models
2. **Path Traversal**: Sanitize all file paths from user input
3. **Resource Limits**: Max context size, max model size, timeout limits
4. **Sandboxing**: No code execution (initially), read-only file access
5. **Privacy**: All processing local, no telemetry by default

## Performance Targets

- **Startup Time**: < 2s (without model load), < 10s (with model load)
- **First Token**: < 5s after query submission
- **Tokens/sec**: 5-20 tokens/sec (CPU dependent)
- **Context Building**: < 1s for projects up to 10k files
- **Memory Overhead**: < 500MB (excluding model weights)

## Extensibility Points

1. **Model Backends**: Interface for different inference engines
2. **Context Providers**: Plugin system for custom indexing
3. **Output Formatters**: Pluggable response rendering
4. **Prompt Templates**: User-customizable templates
5. **Commands**: Cobra subcommand registration

## Status

- **Current**: Design phase
- **Last Updated**: 2026-01-30
- **Implementation**: See PLAN.md for progress
