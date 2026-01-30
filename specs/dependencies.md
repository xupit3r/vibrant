# Dependencies Specification

## Core Dependencies

### CLI Framework
**Package**: `github.com/spf13/cobra`  
**Version**: v1.8.0+  
**License**: Apache 2.0  
**Purpose**: Command-line interface structure and argument parsing  
**Rationale**: Industry standard for Go CLI apps, excellent documentation, wide adoption  

### TUI Framework
**Package**: `github.com/charmbracelet/bubbletea`  
**Version**: v0.25.0+  
**License**: MIT  
**Purpose**: Interactive terminal UI for chat mode  
**Rationale**: Modern, composable TUI framework with excellent tea ecosystem  

**Related Packages**:
- `github.com/charmbracelet/lipgloss` - Styling and layout
- `github.com/charmbracelet/bubbles` - Pre-built UI components

### Configuration Management
**Package**: `github.com/spf13/viper`  
**Version**: v1.18.0+  
**License**: MIT  
**Purpose**: Configuration file, environment variable, and flag management  
**Rationale**: Pairs well with Cobra, handles multiple config sources, widely used  

### LLM Inference
**Package**: `github.com/go-skynet/go-llama.cpp` OR `github.com/ggerganov/llama.cpp` (CGO)  
**Version**: Latest compatible with llama.cpp  
**License**: MIT  
**Purpose**: GGUF model loading and CPU inference  
**Rationale**: Official llama.cpp bindings, best performance, supports latest formats  
**Build Requirements**: CGO enabled, C++ compiler  

**Alternative**: Consider evaluating both go-skynet wrapper vs direct CGO bindings

### Logging
**Package**: `github.com/sirupsen/logrus` OR `go.uber.org/zap`  
**Version**: Latest  
**License**: MIT  
**Purpose**: Structured logging with levels and formatting  
**Rationale**:
- **logrus**: Simple, widely used, good for general purpose
- **zap**: Higher performance, structured, better for high-throughput

**Recommendation**: Start with logrus for simplicity

### File System Operations
**Package**: `github.com/karrick/godirwalk`  
**Version**: v1.17.0+  
**License**: BSD-2-Clause  
**Purpose**: Fast directory traversal for context indexing  
**Rationale**: Much faster than filepath.Walk for large directories  

### Gitignore Parsing
**Package**: `github.com/sabhiram/go-gitignore`  
**Version**: Latest  
**License**: MIT  
**Purpose**: Parse .gitignore files for context filtering  
**Rationale**: Accurate gitignore implementation, handles edge cases  

### HTTP Client (for downloads)
**Package**: Standard library `net/http` + `github.com/schollz/progressbar/v3`  
**Version**: v3.14.0+  
**License**: MIT  
**Purpose**: Model downloads with progress indication  
**Rationale**: Built-in HTTP client is sufficient, progressbar adds UX  

### Syntax Highlighting
**Package**: `github.com/alecthomas/chroma`  
**Version**: v2.12.0+  
**License**: MIT  
**Purpose**: Code syntax highlighting in terminal output  
**Rationale**: Comprehensive language support, TextMate grammar compatible  

### YAML Parsing
**Package**: `gopkg.in/yaml.v3`  
**Version**: v3.0.1+  
**License**: Apache 2.0 / MIT  
**Purpose**: Config file parsing  
**Rationale**: Standard YAML library for Go, integrated with Viper  

## Optional Dependencies

### Token Counting
**Package**: `github.com/pkoukk/tiktoken-go`  
**Version**: Latest  
**License**: MIT  
**Purpose**: Estimate token counts for context management  
**Rationale**: Useful for accurate context window management  
**Note**: May need model-specific implementations

### Embeddings (Future - RAG)
**Package**: TBD - `github.com/tmc/langchaingo` or custom  
**Version**: Latest  
**License**: MIT  
**Purpose**: Generate embeddings for semantic code search  
**Rationale**: Enable RAG for large codebases  
**Status**: Phase 7 (optional)

### Vector Store (Future - RAG)
**Package**: `github.com/chroma-go/chroma-go` OR in-memory  
**Version**: Latest  
**License**: Apache 2.0  
**Purpose**: Store and query code embeddings  
**Rationale**: Local vector database for semantic search  
**Status**: Phase 7 (optional)

## System Dependencies

### llama.cpp
**Repository**: https://github.com/ggerganov/llama.cpp  
**Version**: Latest stable (v0.2.0+)  
**Purpose**: Core inference engine (via CGO)  
**Build Requirements**:
- C++11 compiler (gcc 8+, clang 10+, MSVC 2019+)
- CMake 3.14+ (if building from source)
- Make

### Platform-Specific

#### Linux
- `gcc` or `clang`
- `pkg-config`
- Standard C/C++ libraries

#### macOS
- Xcode Command Line Tools
- Or: `brew install gcc` (optional)

#### Windows
- MinGW-w64 or MSVC
- CGO_ENABLED=1 environment variable

## Development Dependencies

### Testing
**Package**: Standard library `testing` + `github.com/stretchr/testify`  
**Version**: v1.9.0+  
**License**: MIT  
**Purpose**: Unit and integration testing  
**Rationale**: testify adds assertions and mocking capabilities  

### Code Quality
**Tool**: `golangci-lint`  
**Version**: Latest  
**Purpose**: Linting and static analysis  
**Rationale**: Meta-linter combining multiple linters  

### Documentation
**Tool**: `godoc` or `pkgsite`  
**Purpose**: Generate documentation from comments  

## Build Configuration

### go.mod (Initial)

```go
module github.com/joe/vibrant

go 1.22

require (
    github.com/spf13/cobra v1.8.0
    github.com/spf13/viper v1.18.0
    github.com/charmbracelet/bubbletea v0.25.0
    github.com/charmbracelet/lipgloss v0.9.1
    github.com/charmbracelet/bubbles v0.18.0
    github.com/sirupsen/logrus v1.9.3
    github.com/karrick/godirwalk v1.17.0
    github.com/sabhiram/go-gitignore v0.0.0-20210923224102-525f6e181f06
    github.com/schollz/progressbar/v3 v3.14.1
    github.com/alecthomas/chroma v2.12.0
    gopkg.in/yaml.v3 v3.0.1
)

// Optional - will add when implementing
// github.com/pkoukk/tiktoken-go
// github.com/go-skynet/go-llama.cpp
```

### Build Flags

```makefile
# CGO required for llama.cpp
CGO_ENABLED=1

# Build with static linking (optional)
CGO_LDFLAGS="-static"

# Release build (optimized)
go build -ldflags="-s -w" -o vibrant ./cmd/vibrant
```

## Dependency Management Strategy

### Update Policy
- **Major versions**: Manual review, test thoroughly
- **Minor/Patch**: Auto-update monthly (via Dependabot)
- **Security updates**: Immediate

### Vendoring
- **Decision**: Use Go modules only (no vendoring initially)
- **Rationale**: Simplifies development, modules are reliable
- **Future**: Consider vendoring for release builds if needed

### Binary Size Optimization
- Strip debug symbols: `-ldflags="-s -w"`
- Use UPX compression (optional): `upx --best vibrant`
- Exclude unused code via build tags

## Licensing Compliance

All dependencies use permissive licenses compatible with commercial use:
- MIT License: Most dependencies
- Apache 2.0: Cobra, some utilities
- BSD-2-Clause: godirwalk

**Action Required**: Include LICENSE and NOTICE files in distributions

## Installation Requirements

### For Users (Binary Distribution)
- No dependencies (static binary)
- macOS: Gatekeeper may require security exception

### For Developers (Building from Source)
```bash
# Install Go 1.22+
# Install build tools
sudo apt-get install build-essential  # Linux
xcode-select --install                # macOS

# Clone and build
git clone https://github.com/joe/vibrant
cd vibrant
make build  # or: go build ./cmd/vibrant
```

## Dependency Graph (Simplified)

```
vibrant
├── cobra (CLI)
│   └── viper (config)
├── bubbletea (TUI)
│   ├── lipgloss (styling)
│   └── bubbles (components)
├── llama.cpp (inference) [CGO]
├── logrus (logging)
├── godirwalk (file indexing)
├── go-gitignore (filtering)
├── progressbar (downloads)
├── chroma (syntax highlighting)
└── yaml.v3 (config parsing)
```

## Risk Assessment

### High Risk
- **llama.cpp CGO dependency**: Platform-specific builds, complex setup
  - **Mitigation**: Provide pre-built binaries, Docker images
  
### Medium Risk
- **Build complexity**: CGO requires C++ toolchain
  - **Mitigation**: Clear documentation, CI/CD automation
  
### Low Risk
- **Pure Go dependencies**: Generally stable and cross-platform

## Status

- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Next Steps**: Initialize go.mod and add initial dependencies in Phase 1
