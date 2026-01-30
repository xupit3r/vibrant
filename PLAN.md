# Vibrant - Local LLM Code Assistant - Implementation Plan

## Problem Statement
Create a standalone CLI-based code assistant called **Vibrant** that:
- Runs locally using CPU-optimized LLM models
- Provides context-aware coding assistance
- Auto-detects system capabilities and selects appropriate model
- Built in Go for lightweight performance
- Maintains detailed specification documentation in sync with implementation

## Proposed Approach

### Technology Stack
- **Language**: Go (lightweight, excellent concurrency, cross-platform)
- **LLM Backend**: llama.cpp via Go bindings (go-llama.cpp or similar)
- **Model Format**: GGUF (quantized models optimized for CPU inference)
- **Recommended Models**:
  - Qwen 2.5 Coder series (3B/7B/14B)
  - DeepSeek Coder series (1.3B/6.7B)
  - CodeLlama series (7B/13B)

### Architecture
```
┌─────────────────────────────────────┐
│         CLI Interface               │
│  (cobra/bubbletea for rich TUI)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Assistant Core Engine          │
│  - Conversation Manager              │
│  - Context Builder                   │
│  - Model Manager                     │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌───▼────┐
│ Model │  │ Code │  │ Config │
│ Layer │  │ Index│  │ Manager│
└───────┘  └──────┘  └────────┘
```

## Work Plan

### Phase 1: Project Setup & Foundation
- [x] Create project directory structure (`/home/joe/code/vibrant`)
- [x] Write plan to project
- [ ] Initialize Go module with `go mod init github.com/yourusername/vibrant`
- [ ] Create `specs/` directory for technical specifications
- [ ] Create initial spec files (see Specification Documentation section below)
- [ ] Set up basic CLI framework using cobra
- [ ] Create configuration system (YAML/TOML for settings)
- [ ] Implement logging infrastructure
- [ ] Update specs as each component is implemented

### Phase 2: System Detection & Model Management
- [ ] Implement RAM detection utility
- [ ] Create model registry (metadata for supported models)
- [ ] Build model selection algorithm based on available RAM
- [ ] Implement model downloader (from Hugging Face)
- [ ] Create model cache/storage management
- [ ] Update `specs/model-management.md` with implementation details

### Phase 3: LLM Integration
- [ ] Integrate llama.cpp Go bindings
- [ ] Implement model loader with GGUF support
- [ ] Create inference wrapper with streaming support
- [ ] Add quantization support (Q4_K_M, Q5_K_M, Q8_0)
- [ ] Implement context window management
- [ ] Update `specs/llm-integration.md` with API details

### Phase 4: Code Context System
- [ ] Build file tree walker/indexer
- [ ] Implement .gitignore-aware file filtering
- [ ] Create code chunking/embedding system (optional: local embeddings)
- [ ] Build context retrieval (semantic or keyword-based)
- [ ] Implement project structure summarization
- [ ] Update `specs/context-system.md` with indexing algorithms

### Phase 5: Assistant Core Features
- [ ] Conversation history manager (in-memory + optional persistence)
- [ ] Prompt template system for coding tasks
- [ ] Context injection (file contents, project structure)
- [ ] Response streaming to terminal
- [ ] Error handling and recovery
- [ ] Update `specs/assistant-core.md` with conversation flow

### Phase 6: CLI User Experience
- [ ] Interactive chat mode (bubbletea TUI)
- [ ] Single-shot query mode (`vibrant ask "question"`)
- [ ] File/directory context passing (`--context ./src`)
- [ ] Syntax highlighting for code blocks
- [ ] Copy/save response functionality
- [ ] Update `specs/cli-interface.md` with commands and UX flows

### Phase 7: Advanced Features (Optional)
- [ ] Multi-turn conversation with context pruning
- [ ] Code diff generation and application
- [ ] Integration with git for commit message generation
- [ ] RAG (Retrieval Augmented Generation) for large codebases
- [ ] Plugin system for extensibility

### Phase 8: Testing & Optimization
- [ ] Unit tests for core components
- [ ] Integration tests with sample models
- [ ] Performance profiling and optimization
- [ ] Memory usage optimization
- [ ] Documentation and README
- [ ] Final spec review and sync with implementation

## Technical Considerations

### Performance
- Use goroutines for concurrent operations (file indexing, streaming)
- Implement lazy loading for models
- Cache frequently accessed code contexts
- Use memory-mapped files for large models (llama.cpp handles this)

### Model Selection Logic
```
Available RAM → Model Size
< 8 GB       → 3B models (Q4_K_M)
8-16 GB      → 7B models (Q4_K_M or Q5_K_M)
16-32 GB     → 14B models (Q5_K_M)
> 32 GB      → 14B+ models (Q8_0)
```

### Code Context Strategy
1. Always include: Project README, file tree structure
2. Intelligently include: Related files based on query
3. Keep context under model's limit (typically 4K-32K tokens)
4. Priority order: Current file → Imports → Related files

### Dependencies (Estimated)
- `github.com/spf13/cobra` - CLI framework
- `github.com/charmbracelet/bubbletea` - TUI framework
- Go llama.cpp bindings (e.g., `github.com/go-skynet/go-llama.cpp`)
- `github.com/spf13/viper` - Configuration management
- `github.com/chroma-go` or similar - Optional: vector embeddings

## Success Criteria
- [ ] Successfully loads and runs code-optimized models on CPU
- [ ] Responds to coding queries with context awareness
- [ ] Auto-selects appropriate model based on system RAM
- [ ] Provides good user experience in terminal
- [ ] Reasonable response times (< 30s for typical queries)
- [ ] Memory usage stays within system constraints

## Future Enhancements
- Web UI mode
- VSCode extension integration
- Support for multiple concurrent model instances
- Fine-tuning support for domain-specific code
- Team/shared model cache
- Cloud model fallback option

## Specification Documentation

The `specs/` directory will contain detailed technical specifications for each major component. These files must be kept in sync with the implementation at all times.

### Required Spec Files

1. **`specs/architecture.md`**
   - Overall system architecture
   - Component interaction diagrams
   - Data flow diagrams
   - Directory structure and organization

2. **`specs/model-management.md`**
   - Model registry format and structure
   - Download and caching mechanisms
   - Model selection algorithm (RAM-based)
   - Supported model formats and quantizations
   - Model metadata schema

3. **`specs/llm-integration.md`**
   - llama.cpp bindings integration details
   - Inference API and streaming protocol
   - Context window management strategy
   - Token counting and estimation
   - Error handling and recovery

4. **`specs/context-system.md`**
   - File indexing algorithm
   - Gitignore parsing and filtering
   - Context retrieval strategies
   - Chunking and relevance scoring
   - Project structure summarization

5. **`specs/assistant-core.md`**
   - Conversation manager design
   - Prompt template system
   - Context injection pipeline
   - History management and pruning
   - Session persistence

6. **`specs/cli-interface.md`**
   - Command structure and subcommands
   - Interactive mode (bubbletea) design
   - Single-shot mode interface
   - Configuration file format
   - Output formatting and styling

7. **`specs/configuration.md`**
   - Configuration file schema (YAML/TOML)
   - Environment variables
   - Default values and overrides
   - User preferences management

8. **`specs/dependencies.md`**
   - Complete dependency list with versions
   - Rationale for each library choice
   - Licensing information
   - Build requirements

### Spec Maintenance Process
- Create initial spec files during Phase 1
- Update specs **before** implementing new features (design first)
- Review and sync specs **after** implementation (capture reality)
- Include code examples and API signatures in specs
- Add diagrams where helpful (ASCII art, mermaid, etc.)

## Notes
- Focus on CPU inference - avoid CUDA/GPU dependencies
- Prioritize small, efficient models for better response times
- Consider supporting both local models and ollama integration
- Keep configuration simple (sensible defaults)
- Make it easy to switch models without reconfiguration
- **Specs are living documents** - update them as the code evolves
