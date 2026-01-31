# Setting Up llama.cpp for Vibrant

## Overview

Vibrant uses llama.cpp for local LLM inference. By default, the build system attempts to compile with llama.cpp support and falls back to a mock engine if the dependencies aren't available.

## Quick Start

The easiest way to set up dependencies is using the automated script:

```bash
# Check what dependencies are missing
./scripts/check-deps.sh

# Or use make target
make check-deps

# Install missing dependencies automatically
./scripts/check-deps.sh --install

# Or use make target
make install-deps

# Try building (will use llama.cpp if available, mock otherwise)
make build
```

### Script Options

```bash
./scripts/check-deps.sh [OPTIONS]

OPTIONS:
    --install       Install missing dependencies (requires sudo on Linux)
    --verbose       Show detailed output including tool locations
    --dry-run       Show what would be installed without installing
    --quiet         Minimal output (errors only)
    --help          Show help message
    --version       Show version information
```

## Prerequisites

### Linux (Ubuntu/Debian)

```bash
# Install C++ compiler and build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Install Go 1.21+
# (if not already installed)
```

### Linux (Fedora/RHEL)

```bash
# Install C++ compiler and build tools
sudo dnf install -y gcc-c++ cmake git

# Install Go 1.21+
# (if not already installed)
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Or install via Homebrew
brew install cmake

# Install Go 1.21+
brew install go
```

### Windows

1. Install [Visual Studio 2019+](https://visualstudio.microsoft.com/) with C++ support
2. Install [CMake](https://cmake.org/download/)
3. Install [Go 1.21+](https://golang.org/dl/)
4. Use Developer Command Prompt or PowerShell with appropriate environment

## Building with llama.cpp

### Option 1: Automatic (Recommended)

```bash
make build
```

This will:
1. Try to build with llama.cpp support
2. Fall back to mock engine if llama.cpp isn't available
3. Show helpful error messages

### Option 2: Force llama.cpp Build

```bash
make build-llama
```

This will:
1. Attempt to build with llama.cpp
2. Fail with error if dependencies are missing
3. Useful for CI/production builds

### Option 3: Manual Build

```bash
# Enable CGO and set build tag
CGO_ENABLED=1 go build -tags llama -o vibrant ./cmd/vibrant
```

## Verifying Your Build

### Check Build Type

Run vibrant and look at the response from a model:

```bash
# With mock engine, you'll see:
./vibrant ask "test"
# Output contains: "[MOCK RESPONSE - llama.cpp not compiled]"

# With llama.cpp, you'll get real inference:
./vibrant ask "test"
# Output is actual model response
```

### Run Tests

```bash
# Run all tests (works with both build types)
go test ./...

# Run LLM-specific tests
go test ./internal/llm -v

# Check which build is active
go test ./internal/llm -v -run TestBuildTag
```

## Troubleshooting

## Troubleshooting

### Error: "common.h: No such file or directory"

**Problem**: The go-llama.cpp bindings require the llama.cpp source code as a git submodule, which isn't automatically downloaded by Go modules.

**Root Cause**: 
- go-llama.cpp uses git submodules to include llama.cpp source
- Go module cache is read-only and doesn't clone submodules
- The C++ bindings need llama.cpp headers and source files

**Solutions**:

**Option 1: Use Mock Engine (Recommended for Development)**
```bash
make build-mock
```
The mock engine works perfectly for development and testing. All features work except actual LLM inference.

**Option 2: Setup Check Script**
```bash
./scripts/setup-llama.sh
```
This script explains the issue and provides detailed setup instructions.

**Option 3: Manual Vendor Setup**
```bash
# Clone go-llama.cpp with submodules into vendor directory
mkdir -p vendor/github.com/go-skynet
cd vendor/github.com/go-skynet
git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp
cd go-llama.cpp
make libbinding.a

# Return to project root
cd ../../../../

# Build with vendor mode
go build -mod=vendor -tags llama -o vibrant ./cmd/vibrant
```

**Option 4: Use Ollama (Easiest for Real Inference)**

Ollama is a much easier alternative that manages models and provides an API:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Ollama runs as a service and provides an API
# Future versions of Vibrant will support Ollama directly
```

**Why This Happens**:
Go's module system downloads source code but doesn't execute git commands like submodule initialization. The go-llama.cpp library expects llama.cpp source to be in a subdirectory, which requires git submodule init. This is a known limitation of Go modules with C/C++ bindings that use submodules.

**Current Status**:
- Mock engine: ✅ Works out of the box
- Real inference: ⚠️ Requires manual setup
- Future: Will add Ollama support for easier real inference

### Error: "undefined reference to llama_*"

**Problem**: llama.cpp library isn't linked properly.

**Solution**:
```bash
# Ensure CGO is enabled
export CGO_ENABLED=1

# Try building again
make build-llama
```

### Build is Slow

**Problem**: CGO compilation takes time.

**Solutions**:
- Use `make build-mock` for development
- Use `make build-llama` only for testing inference
- Compiled binary can be reused without rebuilding

### Windows-Specific Issues

**Problem**: CGO doesn't work on Windows easily.

**Solutions**:
1. Use WSL2 (Windows Subsystem for Linux)
2. Use Docker to build
3. Use mock engine for development

## Development Workflow

### Recommended Approach

```bash
# Development (fast builds, no inference)
make build-mock

# Testing inference (slower build, real LLM)
make build-llama

# Production deployment
make build  # Tries llama, falls back to mock
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential
      
      - name: Build with llama.cpp
        run: make build-llama
      
      - name: Run tests
        run: go test ./...
```

## Performance Considerations

### CPU vs GPU

Vibrant focuses on **CPU inference** for maximum compatibility. GPU support via llama.cpp is possible but not the primary target.

### Model Selection

Choose models based on available RAM:

| RAM Available | Recommended Model        | Quantization |
|---------------|-------------------------|--------------|
| 8 GB          | Qwen 2.5 Coder 3B       | Q4_K_M       |
| 12 GB         | Qwen 2.5 Coder 7B       | Q4_K_M       |
| 16 GB         | Qwen 2.5 Coder 7B       | Q5_K_M       |
| 24+ GB        | Qwen 2.5 Coder 14B      | Q5_K_M       |

### Optimization Tips

1. **Thread Count**: Default is `runtime.NumCPU()`, adjust if needed
2. **Context Size**: Smaller = faster, but less context
3. **Quantization**: Q4_K_M is fastest, Q8_0 is most accurate
4. **Model Size**: Smaller models respond faster

## Alternative: Using Ollama

If llama.cpp integration is problematic, consider using Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Vibrant can integrate with Ollama (future feature)
```

## Testing Without llama.cpp

The mock engine is fully functional for development:

```bash
# Build with mock
make build-mock

# All commands work, just with mock responses
./vibrant model list
./vibrant ask "explain goroutines"
./vibrant chat

# Tests all pass
go test ./...
```

## Getting Help

If you encounter issues:

1. Check this guide first
2. Try `make build-mock` to isolate the issue
3. Check Go and C++ compiler versions
4. File an issue with:
   - OS and version
   - Go version (`go version`)
   - GCC/Clang version (`gcc --version` or `clang --version`)
   - Full error output

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [go-llama.cpp bindings](https://github.com/go-skynet/go-llama.cpp)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)
- [Vibrant Build System](../Makefile)
