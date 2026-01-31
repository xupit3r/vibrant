.PHONY: build clean test install run help

# Build variables
BINARY_NAME=vibrant
BUILD_DIR=./build
CMD_DIR=./cmd/vibrant
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "v0.1.0-dev")
LDFLAGS=-ldflags "-s -w -X main.version=$(VERSION)"
BUILD_TAGS=-tags llama
CGO_FLAGS=CGO_ENABLED=1

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the binary (tries llama.cpp, falls back to mock if unavailable)
	@echo "Building $(BINARY_NAME)..."
	@if $(CGO_FLAGS) go build $(BUILD_TAGS) $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR) 2>/dev/null; then \
		echo "Build complete: ./$(BINARY_NAME) (with llama.cpp)"; \
	else \
		echo "⚠️  llama.cpp build failed, falling back to mock engine..."; \
		go build $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR); \
		echo "Build complete: ./$(BINARY_NAME) (mock engine)"; \
		echo ""; \
		echo "💡 To enable real inference, install C++ compiler and llama.cpp:"; \
		echo "   See docs/llm-integration.md for setup instructions"; \
	fi

build-llama: ## Build with llama.cpp inference (requires C++ compiler)
	@echo "Building $(BINARY_NAME) with llama.cpp..."
	@$(CGO_FLAGS) go build $(BUILD_TAGS) $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "Build complete: ./$(BINARY_NAME)"

build-mock: ## Build with mock engine (for testing without C++ compiler)
	@echo "Building $(BINARY_NAME) with mock engine..."
	@go build $(LDFLAGS) -o $(BINARY_NAME) $(CMD_DIR)
	@echo "Build complete: ./$(BINARY_NAME) (mock engine)"

build-all: ## Build for all platforms with llama.cpp
	@echo "Building for all platforms with llama.cpp..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build $(BUILD_TAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build $(BUILD_TAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-arm64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build $(BUILD_TAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build $(BUILD_TAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 $(CMD_DIR)
	@CGO_ENABLED=1 GOOS=windows GOARCH=amd64 go build $(BUILD_TAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe $(CMD_DIR)
	@echo "Build complete: $(BUILD_DIR)/"

clean: ## Clean build artifacts
	@echo "Cleaning..."
	@rm -f $(BINARY_NAME)
	@rm -rf $(BUILD_DIR)
	@go clean
	@echo "Clean complete"

test: ## Run tests
	@echo "Running tests..."
	@go test -v ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@go test -v -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

install: build ## Install binary to GOPATH/bin
	@echo "Installing $(BINARY_NAME)..."
	@$(CGO_FLAGS) go install $(BUILD_TAGS) $(LDFLAGS) $(CMD_DIR)
	@echo "Installed to $(GOPATH)/bin/$(BINARY_NAME)"
	@echo ""
	@echo "💡 Don't forget to set up shell completion!"
	@echo "   Zsh:  $(BINARY_NAME) completion zsh > ~/.zsh/completion/_vibrant"
	@echo "   Bash: $(BINARY_NAME) completion bash > ~/.local/share/bash-completion/completions/vibrant"
	@echo "   Fish: $(BINARY_NAME) completion fish > ~/.config/fish/completions/vibrant.fish"
	@echo ""
	@echo "See docs/COMPLETION-QUICKSTART.md for details"

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
	@go test ./test/bench/... -bench=. -benchmem

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
