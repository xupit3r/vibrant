# CLI Interface Specification

## Overview
The CLI interface provides both single-shot and interactive modes for interacting with the Vibrant code assistant. It includes advanced shell tab-completion for a streamlined user experience.

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
├── completion               # Generate shell completion scripts
│   ├── bash                # Generate bash completion
│   ├── zsh                 # Generate zsh completion
│   ├── fish                # Generate fish completion
│   └── powershell          # Generate PowerShell completion
└── version                  # Version information
```

## Shell Tab-Completion

### Overview
Vibrant provides advanced shell tab-completion for zsh, bash, and fish shells. The completion system is **context-aware**, offering intelligent suggestions based on the current command context.

### Features
- **Subcommand completion**: Press Tab after `vibrant` to see all available commands
- **Flag completion**: Shows all available flags for each command
- **Model ID completion**: Suggests model IDs when using `model download`, `model info`, or `model remove`
- **Config key completion**: Suggests configuration keys for `config get` and `config set`
- **File path completion**: Standard file/directory completion for `--context` and other path flags
- **Dynamic completion**: Completions update based on installed models and current configuration

### Supported Shells
- **Zsh**: Full support with descriptions and context-aware completions
- **Bash**: Full support with programmable completion
- **Fish**: Full support with rich descriptions and interactive menus

### Installation

#### Zsh
```bash
# Generate and install completion script
vibrant completion zsh > ~/.zsh/completion/_vibrant

# Or install to system directory (may require sudo)
vibrant completion zsh | sudo tee /usr/local/share/zsh/site-functions/_vibrant

# Add to ~/.zshrc if not already present:
fpath=(~/.zsh/completion $fpath)
autoload -Uz compinit && compinit
```

#### Bash
```bash
# Generate and install completion script
vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant

# Or install to system directory (may require sudo)
vibrant completion bash | sudo tee /etc/bash_completion.d/vibrant

# Add to ~/.bashrc if needed:
source ~/.local/share/bash-completion/completions/vibrant
```

#### Fish
```bash
# Generate and install completion script
vibrant completion fish > ~/.config/fish/completions/vibrant.fish

# Or install to system directory (may require sudo)
vibrant completion fish | sudo tee /usr/share/fish/vendor_completions.d/vibrant.fish
```

### Completion Examples

```bash
# List all commands
$ vibrant <TAB>
ask      chat     completion     config     model      version

# Show flags for ask command
$ vibrant ask --<TAB>
--context      # Specify context files/directories
--model        # Specify model to use
--save         # Save response to file
--stream       # Enable streaming mode
--help         # Show help

# Complete model IDs
$ vibrant model download <TAB>
qwen2.5-coder-3b-q4    # Qwen 2.5 Coder 3B (Q4_K_M quantization)
qwen2.5-coder-7b-q4    # Qwen 2.5 Coder 7B (Q4_K_M quantization)
qwen2.5-coder-7b-q5    # Qwen 2.5 Coder 7B (Q5_K_M quantization)
qwen2.5-coder-14b-q5   # Qwen 2.5 Coder 14B (Q5_K_M quantization)

# Complete config keys
$ vibrant config get <TAB>
default_model          # Default model ID
stream_mode            # Enable streaming by default
max_context_size       # Maximum context size in tokens
context_window         # Context window size
temperature            # Sampling temperature
```

### Implementation Details

#### Cobra Framework Integration
Vibrant uses the Cobra CLI framework, which includes built-in support for generating shell completion scripts via the `completion` command.

#### Context-Aware Completions
The completion system provides intelligent suggestions based on:
1. **Current command path**: Different completions for `vibrant model` vs `vibrant config`
2. **Flag values**: Suggests model IDs for `--model` flag
3. **Subcommand arguments**: Suggests appropriate values for positional arguments
4. **System state**: Dynamically includes installed models and current config keys

#### Completion Functions
Custom completion functions are registered for:
- `__vibrant_get_models`: Lists available model IDs from the model registry
- `__vibrant_get_config_keys`: Lists configuration keys from the config system
- `__vibrant_complete_files`: Completes file paths with filtering

### Testing Completion

After installing completion scripts:
```bash
# Test basic completion
vibrant <TAB>

# Test flag completion
vibrant ask --<TAB>

# Test model completion
vibrant model download <TAB>

# Test config completion
vibrant config get <TAB>

# Test file path completion
vibrant ask --context <TAB>
```

### Troubleshooting

**Completion not working:**
1. Ensure completion script is installed in the correct directory
2. Verify your shell configuration sources the completion
3. Reload shell: `exec $SHELL` or restart terminal
4. Check completion is enabled: `compinit` (zsh) or `complete -p vibrant` (bash)

**Outdated completions:**
1. Regenerate completion script: `vibrant completion <shell>`
2. Reinstall to appropriate directory
3. Reload shell

### Future Enhancements
- **PowerShell support**: Add PowerShell completion for Windows users
- **Dynamic model suggestions**: Filter models by available RAM
- **Recent file suggestions**: Suggest recently accessed files for `--context`
- **Completion caching**: Cache expensive completions for better performance

## Status
- **Current**: Full implementation with advanced shell completion
- **Last Updated**: 2026-01-31
- **Implementation**: Phase 6 (complete)
