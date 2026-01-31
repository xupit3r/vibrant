# Shell Tab-Completion Guide

## Overview

Vibrant provides advanced shell tab-completion for zsh, bash, and fish shells. The completion system is context-aware and provides intelligent suggestions based on the current command context.

## Features

- **Subcommand completion**: Press Tab after `vibrant` to see all available commands
- **Flag completion**: Shows all available flags for each command  
- **Model ID completion**: Suggests model IDs when using `model download`, `model info`, or `model remove`
- **Config key completion**: Suggests configuration keys for `config get` and `config set`
- **File path completion**: Standard file/directory completion for `--context` and other path flags
- **Dynamic completion**: Completions update based on installed models and current configuration

## Installation

### Prerequisites

Vibrant must be installed and accessible in your PATH. Build and install with:

```bash
cd /path/to/vibrant
make build
make install
```

### Zsh

```bash
# Create completion directory if it doesn't exist
mkdir -p ~/.zsh/completion

# Generate completion script
vibrant completion zsh > ~/.zsh/completion/_vibrant

# Add to ~/.zshrc (if not already present):
fpath=(~/.zsh/completion $fpath)
autoload -Uz compinit && compinit

# Reload shell
exec zsh
```

**System-wide installation (requires sudo):**
```bash
vibrant completion zsh | sudo tee /usr/local/share/zsh/site-functions/_vibrant
```

### Bash

```bash
# Create completion directory if it doesn't exist
mkdir -p ~/.local/share/bash-completion/completions

# Generate completion script
vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant

# Add to ~/.bashrc (if needed):
if [ -f ~/.local/share/bash-completion/completions/vibrant ]; then
    source ~/.local/share/bash-completion/completions/vibrant
fi

# Reload shell
exec bash
```

**System-wide installation (requires sudo):**
```bash
vibrant completion bash | sudo tee /etc/bash_completion.d/vibrant
```

### Fish

```bash
# Create completion directory if it doesn't exist
mkdir -p ~/.config/fish/completions

# Generate completion script
vibrant completion fish > ~/.config/fish/completions/vibrant.fish

# Reload completions
source ~/.config/fish/completions/vibrant.fish
```

**System-wide installation (requires sudo):**
```bash
vibrant completion fish | sudo tee /usr/share/fish/vendor_completions.d/vibrant.fish
```

### PowerShell (Windows)

```powershell
# Generate and execute completion script (temporary)
vibrant completion powershell | Out-String | Invoke-Expression

# To persist, add to your PowerShell profile:
vibrant completion powershell >> $PROFILE
```

## Usage Examples

### Basic Command Completion

```bash
# Show all available commands
$ vibrant <TAB>
ask         chat        completion  model       version

# Show model subcommands
$ vibrant model <TAB>
download    info        list        remove
```

### Flag Completion

```bash
# Show flags for ask command
$ vibrant ask --<TAB>
--context      Specify context files/directories
--help         Show help for this command
--model        Specify model to use
--save         Save response to file
--stream       Enable streaming mode
--verbose      Verbose output
```

### Model ID Completion

```bash
# Complete model IDs for download
$ vibrant model download <TAB>
qwen2.5-coder-3b-q4     Qwen 2.5 Coder 3B (Q4_K_M quantization) - 4GB RAM
qwen2.5-coder-7b-q4     Qwen 2.5 Coder 7B (Q4_K_M quantization) - 8GB RAM
qwen2.5-coder-7b-q5     Qwen 2.5 Coder 7B (Q5_K_M quantization) - 10GB RAM
qwen2.5-coder-14b-q5    Qwen 2.5 Coder 14B (Q5_K_M quantization) - 18GB RAM

# Complete for info command
$ vibrant model info qwen<TAB>
qwen2.5-coder-3b-q4
qwen2.5-coder-7b-q4
qwen2.5-coder-7b-q5
qwen2.5-coder-14b-q5
```

### Config Key Completion

```bash
# Complete config keys
$ vibrant config get <TAB>
default_model          Default model ID to use
stream_mode            Enable streaming mode by default
max_context_size       Maximum context size in tokens
temperature            Sampling temperature (0.0-2.0)
```

### File Path Completion

```bash
# Complete file paths for context flag
$ vibrant ask --context ./sr<TAB>
./src/

$ vibrant ask --context ./src/main.<TAB>
./src/main.go
```

## Troubleshooting

### Completion Not Working

**1. Verify completion is installed:**

```bash
# Zsh
ls ~/.zsh/completion/_vibrant
# or
ls /usr/local/share/zsh/site-functions/_vibrant

# Bash  
ls ~/.local/share/bash-completion/completions/vibrant
# or
ls /etc/bash_completion.d/vibrant

# Fish
ls ~/.config/fish/completions/vibrant.fish
# or
ls /usr/share/fish/vendor_completions.d/vibrant.fish
```

**2. Check shell configuration:**

```bash
# Zsh - verify fpath includes completion directory
echo $fpath

# Bash - verify completion is sourced
complete -p vibrant

# Fish - list completions
complete -c vibrant
```

**3. Reload shell:**

```bash
# Zsh
exec zsh

# Bash
exec bash

# Fish
exec fish
```

**4. Regenerate completion:**

```bash
# Delete old completion and regenerate
rm ~/.zsh/completion/_vibrant
vibrant completion zsh > ~/.zsh/completion/_vibrant
exec zsh
```

### Completion is Outdated

If you've updated Vibrant and completion suggestions are outdated:

```bash
# Regenerate for your shell (example for zsh)
vibrant completion zsh > ~/.zsh/completion/_vibrant
exec zsh
```

### Partial Completion

If completion only works for some commands:

1. Ensure you're using the latest completion script
2. Check that all subcommands are registered in `completion.go`
3. Verify custom completion functions are registered in `init()`

## Advanced Configuration

### Zsh Completion Options

Add to `~/.zshrc` for enhanced completion:

```zsh
# Case-insensitive completion
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'

# Show completion menu
zstyle ':completion:*' menu select

# Color completion
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"

# Group completions
zstyle ':completion:*' group-name ''
zstyle ':completion:*:descriptions' format '%B%d%b'
```

### Bash Completion Options

Add to `~/.bashrc` for enhanced completion:

```bash
# Case-insensitive completion
bind "set completion-ignore-case on"

# Show all completions immediately
bind "set show-all-if-ambiguous on"
```

### Fish Completion Options

Fish completions are configured per-command in the completion script.

## Development

### Adding New Completions

To add completions for a new command:

1. **Register ValidArgsFunction** in `completion.go`:

```go
yourCmd.ValidArgsFunction = func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
    // Return completion suggestions
    return []string{"option1\tDescription", "option2\tDescription"}, cobra.ShellCompDirectiveNoFileComp
}
```

2. **Test the completion**:

```bash
# Build
make build

# Test manually
./vibrant your-command <TAB>

# Regenerate completion script
./vibrant completion bash > /tmp/test-completion
source /tmp/test-completion
./vibrant your-command <TAB>
```

3. **Add tests** in `completion_test.go`:

```go
func TestYourCommandCompletion(t *testing.T) {
    completions, directive := yourCmd.ValidArgsFunction(yourCmd, []string{}, "")
    // Verify completions
}
```

## References

- [Cobra Shell Completions](https://github.com/spf13/cobra/blob/master/shell_completions.md)
- [Bash Completion Guide](https://github.com/scop/bash-completion)
- [Zsh Completion Guide](https://zsh.sourceforge.io/Doc/Release/Completion-System.html)
- [Fish Completion Guide](https://fishshell.com/docs/current/completions.html)

## Support

For issues or feature requests related to shell completion:
1. Check this guide and the troubleshooting section
2. Regenerate completion scripts with the latest version
3. File an issue on the Vibrant repository with:
   - Shell type and version
   - Output of `vibrant completion <shell>`
   - Steps to reproduce the problem
