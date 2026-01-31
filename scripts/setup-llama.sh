#!/usr/bin/env bash
# Setup script for llama.cpp integration
# Initializes the llama.cpp submodule in go-llama.cpp

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1" >&2
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

echo "========================================="
echo "Vibrant llama.cpp Setup"
echo "========================================="
echo ""

# Check if Go is installed
if ! command -v go >/dev/null 2>&1; then
    log_error "Go is not installed"
    log_info "Install Go from: https://golang.org/dl/"
    exit 1
fi

log_info "Finding go-llama.cpp module..."

# Get the module path
MODULE_PATH=$(go list -m -f '{{.Dir}}' github.com/go-skynet/go-llama.cpp 2>/dev/null)

if [ -z "$MODULE_PATH" ]; then
    log_error "go-llama.cpp module not found"
    log_info "Run 'go mod download' first"
    exit 1
fi

log_success "Found module at: $MODULE_PATH"

# Check if llama.cpp subdirectory exists
if [ -d "$MODULE_PATH/llama.cpp" ] && [ -f "$MODULE_PATH/llama.cpp/llama.h" ]; then
    log_success "llama.cpp is already initialized"
    log_info "You can now build with: make build-llama"
    exit 0
fi

log_warning "llama.cpp submodule not initialized"
echo ""

# The module directory is read-only, so we need to clone our own copy
log_info "The go-llama.cpp module needs manual setup because:"
echo "  1. It uses git submodules (llama.cpp)"
echo "  2. Go module cache is read-only"
echo "  3. We need to clone and build it separately"
echo ""

log_info "Recommended solutions:"
echo ""
echo "Option 1: Use the mock engine (default)"
echo "  make build-mock"
echo ""
echo "Option 2: Clone go-llama.cpp manually"
echo "  mkdir -p vendor/github.com/go-skynet"
echo "  cd vendor/github.com/go-skynet"
echo "  git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp"
echo "  cd go-llama.cpp"
echo "  make libbinding.a"
echo "  cd ../../../../"
echo "  # Then use vendor mode"
echo "  go build -mod=vendor -tags llama ..."
echo ""
echo "Option 3: Use a pre-built llama.cpp binding"
echo "  # See docs/llama-setup.md for alternatives"
echo ""
echo "Option 4: Use Ollama (easier alternative)"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull qwen2.5-coder:7b"
echo "  # Future: Vibrant will support Ollama directly"
echo ""

log_warning "For now, Vibrant uses a mock engine by default"
log_info "This is fine for development and testing"
log_info "Build with: make build (will fallback to mock automatically)"

exit 1
