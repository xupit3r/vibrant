.PHONY: build clean test install run help

# Build variables
BINARY_NAME=vibrant
BUILD_DIR=./build
CMD_DIR=./cmd/vibrant
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "v0.1.0-dev")
LDFLAGS=-ldflags "-s -w -X main.version=$(VERSION)"

# Engine build configurations
# Default: Pure Go custom engine (no CGO required)
BUILD_TAGS_CUSTOM=
CGO_FLAGS_CUSTOM=CGO_ENABLED=0

# Alternative: llama.cpp engine (requires CGO)
BUILD_TAGS_LLAMA=-tags llama
CGO_FLAGS_LLAMA=CGO_ENABLED=1

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the binary with custom pure Go engine (default, no CGO required)
	@echo "Building $(BINARY_NAME) with custom pure Go engine..."
	@$(CGO_FLAGS_CUSTOM) go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (pure Go, no CGO)"
	@echo ""
	@echo "💡 This build uses our custom pure Go inference engine"
	@echo "   For llama.cpp engine: make build-llama"

build-custom: ## Build with custom pure Go engine (same as default build)
	@echo "Building $(BINARY_NAME) with custom pure Go engine..."
	@$(CGO_FLAGS_CUSTOM) go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (pure Go)"

build-gpu: ## Build with GPU support (requires macOS and CGO for Metal)
	@echo "Building $(BINARY_NAME) with GPU support (Metal)..."
	@CGO_ENABLED=1 go build $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (with Metal GPU support)"
	@echo ""
	@echo "💡 This build includes Metal GPU acceleration for Apple Silicon"
	@echo "   Use --device gpu flag to enable GPU acceleration"

build-cuda: ## Build with CUDA GPU support (requires Linux, NVIDIA GPU, and CUDA toolkit)
	@echo "Building $(BINARY_NAME) with CUDA GPU support..."
	@CGO_ENABLED=1 go build $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (with CUDA GPU support)"
	@echo ""
	@echo "💡 This build includes CUDA GPU acceleration for NVIDIA GPUs"
	@echo "   Requires: CUDA Toolkit 12.0+, NVIDIA Driver 525.60.13+"
	@echo "   Use --device cuda flag to enable CUDA acceleration"

build-llama: ## Build with llama.cpp inference (requires C++ compiler and CGO)
	@echo "Building $(BINARY_NAME) with llama.cpp..."
	@$(CGO_FLAGS_LLAMA) go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (llama.cpp via CGO)"
	@echo ""
	@echo "💡 To use custom pure Go engine: make build-custom"

build-mock: ## Build with mock engine (for testing without dependencies)
	@echo "Building $(BINARY_NAME) with mock engine..."
	@CGO_ENABLED=0 go build $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "✅ Build complete: ./$(BINARY_NAME) (mock engine)"

build-all: ## Build for all platforms with custom pure Go engine
	@echo "Building for all platforms with custom pure Go engine..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 $(CMD_DIR)
	@CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-arm64 $(CMD_DIR)
	@CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 $(CMD_DIR)
	@CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 $(CMD_DIR)
	@CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build $(BUILD_TAGS_CUSTOM) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe $(CMD_DIR)
	@echo "✅ Build complete: $(BUILD_DIR)/ (pure Go, cross-platform compatible)"

build-all-llama: ## Build for all platforms with llama.cpp (requires cross-compilation toolchain)
	@echo "Building for all platforms with llama.cpp..."
	@echo "⚠️  Warning: Cross-compiling with CGO requires platform-specific toolchains"
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-llama-linux-amd64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-llama-linux-arm64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-llama-darwin-amd64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-llama-darwin-arm64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=windows GOARCH=amd64 go build $(BUILD_TAGS_LLAMA) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-llama-windows-amd64.exe $(CMD_DIR)
	@echo "✅ Build complete: $(BUILD_DIR)/"

clean: ## Clean build artifacts
	@echo "Cleaning..."
	@rm -f $(BINARY_NAME)
	@rm -rf $(BUILD_DIR)
	@go clean
	@echo "Clean complete"

test: ## Run tests
	@echo "Running tests..."
	@CGO_ENABLED=0 go test -v ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@CGO_ENABLED=0 go test -v -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

test-cuda: ## Run CUDA GPU tests (requires CUDA toolkit)
	@echo "Running CUDA GPU tests..."
	@echo "⚠️  Requires: CUDA Toolkit 12.0+, NVIDIA GPU, and Driver 525.60.13+"
	@CGO_ENABLED=1 go test -v ./internal/gpu -run TestCUDA

install: build ## Install binary to GOPATH/bin (with custom pure Go engine)
	@echo "Installing $(BINARY_NAME) with custom pure Go engine..."
	@$(CGO_FLAGS_CUSTOM) go install $(BUILD_TAGS_CUSTOM) $(LDFLAGS) $(CMD_DIR)
	@echo "✅ Installed to $(GOPATH)/bin/$(BINARY_NAME) (pure Go)"
	@echo ""
	@echo "💡 Don't forget to set up shell completion!"
	@echo "   Zsh:  $(BINARY_NAME) completion zsh > ~/.zsh/completion/_vibrant"
	@echo "   Bash: $(BINARY_NAME) completion bash > ~/.local/share/bash-completion/completions/vibrant"
	@echo "   Fish: $(BINARY_NAME) completion fish > ~/.config/fish/completions/vibrant.fish"
	@echo ""
	@echo "See docs/COMPLETION-QUICKSTART.md for details"

install-llama: ## Install binary to GOPATH/bin (with llama.cpp engine)
	@echo "Installing $(BINARY_NAME) with llama.cpp engine..."
	@$(CGO_FLAGS_LLAMA) go install $(BUILD_TAGS_LLAMA) $(LDFLAGS) $(CMD_DIR)
	@echo "✅ Installed to $(GOPATH)/bin/$(BINARY_NAME) (llama.cpp)"

run: build ## Build and run the binary
	@./$(BINARY_NAME)

fmt: ## Format code
	@echo "Formatting code..."
	@go fmt ./...

lint: ## Run linter
	@echo "Running linter..."
	@golangci-lint run || echo "golangci-lint not installed. Run: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@CGO_ENABLED=0 go test ./internal/... -bench=. -benchmem -benchtime=3s

bench-tensor: ## Run tensor benchmarks only
	@echo "Running tensor benchmarks..."
	@CGO_ENABLED=0 go test ./internal/tensor -bench=. -benchmem -benchtime=5s

bench-inference: ## Run inference benchmarks (requires test model)
	@echo "Running inference benchmarks..."
	@CGO_ENABLED=0 go test ./internal/inference -bench=. -benchmem -benchtime=3s

bench-cuda: ## Run CUDA GPU benchmarks (requires CUDA toolkit)
	@echo "Running CUDA GPU benchmarks..."
	@echo "⚠️  Requires: CUDA Toolkit 12.0+, NVIDIA GPU, and Driver 525.60.13+"
	@CGO_ENABLED=1 go test ./internal/gpu -bench=BenchmarkCUDA -benchmem -benchtime=3s

bench-compare: ## Run benchmarks and compare with baseline
	@echo "Running benchmarks and comparing..."
	@go test ./internal/... -bench=. -benchmem -benchtime=3s | tee bench-new.txt
	@if [ -f bench-baseline.txt ]; then \
		benchstat bench-baseline.txt bench-new.txt; \
	else \
		echo "No baseline found. Saving current as baseline..."; \
		cp bench-new.txt bench-baseline.txt; \
	fi

check-deps: ## Check for required dependencies
	@./scripts/check-deps.sh

install-deps: ## Install missing dependencies
	@./scripts/check-deps.sh --install

setup-llama: ## Check llama.cpp setup and show instructions
	@./scripts/setup-llama.sh || true

tidy: ## Tidy dependencies
	@echo "Tidying dependencies..."
	@go mod tidy

deps: ## Download dependencies
	@echo "Downloading dependencies..."
	@go mod download

dev: ## Run in development mode with auto-reload (requires air)
	@air || echo "air not installed. Run: go install github.com/cosmtrek/air@latest"

.DEFAULT_GOAL := help
