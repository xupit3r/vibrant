package commands

import (
	"os"

	"github.com/spf13/cobra"
)

var completionCmd = &cobra.Command{
	Use:   "completion [bash|zsh|fish|powershell]",
	Short: "Generate shell completion script",
	Long: `Generate shell completion script for Vibrant.

To load completions:

Bash:
  $ vibrant completion bash > ~/.local/share/bash-completion/completions/vibrant
  $ source ~/.local/share/bash-completion/completions/vibrant

Zsh:
  $ vibrant completion zsh > ~/.zsh/completion/_vibrant
  $ echo 'fpath=(~/.zsh/completion $fpath)' >> ~/.zshrc
  $ echo 'autoload -Uz compinit && compinit' >> ~/.zshrc

Fish:
  $ vibrant completion fish > ~/.config/fish/completions/vibrant.fish

PowerShell:
  PS> vibrant completion powershell | Out-String | Invoke-Expression
  # To persist, add the output to your PowerShell profile
`,
	DisableFlagsInUseLine: true,
	ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
	Args:                  cobra.ExactValidArgs(1),
	RunE:                  runCompletion,
}

func init() {
	rootCmd.AddCommand(completionCmd)
	
	// Register custom completion functions for various commands
	registerModelCompletions()
	registerConfigCompletions()
}

func runCompletion(cmd *cobra.Command, args []string) error {
	switch args[0] {
	case "bash":
		return cmd.Root().GenBashCompletion(os.Stdout)
	case "zsh":
		return cmd.Root().GenZshCompletion(os.Stdout)
	case "fish":
		return cmd.Root().GenFishCompletion(os.Stdout, true)
	case "powershell":
		return cmd.Root().GenPowerShellCompletionWithDesc(os.Stdout)
	}
	return nil
}

// registerModelCompletions registers custom completions for model commands
func registerModelCompletions() {
	// Model IDs completion for download, info, and remove commands
	validModelIDs := func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) != 0 {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}
		
		// Model IDs from registry
		modelIDs := []string{
			"qwen2.5-coder-3b-q4\tQwen 2.5 Coder 3B (Q4_K_M quantization) - 4GB RAM",
			"qwen2.5-coder-7b-q4\tQwen 2.5 Coder 7B (Q4_K_M quantization) - 8GB RAM",
			"qwen2.5-coder-7b-q5\tQwen 2.5 Coder 7B (Q5_K_M quantization) - 10GB RAM",
			"qwen2.5-coder-14b-q5\tQwen 2.5 Coder 14B (Q5_K_M quantization) - 18GB RAM",
		}
		
		return modelIDs, cobra.ShellCompDirectiveNoFileComp
	}
	
	modelDownloadCmd.ValidArgsFunction = validModelIDs
	modelInfoCmd.ValidArgsFunction = validModelIDs
	modelRemoveCmd.ValidArgsFunction = validModelIDs
}

// registerConfigCompletions registers custom completions for config commands
func registerConfigCompletions() {
	// Config keys completion function
	// This will be used when config commands (get/set) are implemented
	// For now, we define it as a placeholder for future use
	_ = func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) != 0 {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}
		
		// Common config keys
		configKeys := []string{
			"default_model\tDefault model ID to use",
			"stream_mode\tEnable streaming mode by default",
			"max_context_size\tMaximum context size in tokens",
			"context_window\tContext window size for the model",
			"temperature\tSampling temperature (0.0-2.0)",
			"top_p\tNucleus sampling parameter",
			"top_k\tTop-K sampling parameter",
			"max_tokens\tMaximum tokens to generate",
			"verbose\tEnable verbose logging",
			"no_color\tDisable colored output",
		}
		
		return configKeys, cobra.ShellCompDirectiveNoFileComp
	}
	
	// Note: This completion function will be attached to config commands
	// when they are implemented (config get/set commands)
}
