# Shell Completion Quick Start

After installing vibrant, set up tab-completion in 3 steps:

## Zsh (macOS default, Linux)
```bash
vibrant completion zsh > ~/.zsh/completion/_vibrant
echo 'fpath=(~/.zsh/completion $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc
exec zsh
```

## Bash (Linux default)
```bash
vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant
echo 'source ~/.local/share/bash-completion/completions/vibrant' >> ~/.bashrc
exec bash
```

## Fish (Modern shell)
```bash
vibrant completion fish > ~/.config/fish/completions/vibrant.fish
exec fish
```

## Try It Out
```bash
vibrant <TAB>                    # Show commands
vibrant model download <TAB>     # Show models (with RAM requirements!)
vibrant ask --<TAB>              # Show flags
```

See `docs/shell-completion.md` for full documentation.
