package commands

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestCompletionCommand(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{
			name:    "bash completion",
			args:    []string{"completion", "bash"},
			wantErr: false,
		},
		{
			name:    "zsh completion",
			args:    []string{"completion", "zsh"},
			wantErr: false,
		},
		{
			name:    "fish completion",
			args:    []string{"completion", "fish"},
			wantErr: false,
		},
		{
			name:    "powershell completion",
			args:    []string{"completion", "powershell"},
			wantErr: false,
		},
		{
			name:    "invalid shell",
			args:    []string{"completion", "invalid"},
			wantErr: true,
		},
		{
			name:    "no shell specified",
			args:    []string{"completion"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fresh root command that includes completion
			cmd := createTestRootCommand()

			// Set args
			cmd.SetArgs(tt.args)

			// Execute
			err := cmd.Execute()

			// Check error expectation
			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

// createTestRootCommand creates a complete test root command with all subcommands
func createTestRootCommand() *cobra.Command {
	root := &cobra.Command{
		Use:     "vibrant",
		Short:   "A local LLM code assistant",
		Version: "test",
	}

	// Add the actual completion command
	root.AddCommand(completionCmd)

	// Add other essential commands for realistic testing
	root.AddCommand(modelCmd)

	return root
}

func TestModelIDCompletions(t *testing.T) {
	tests := []struct {
		name      string
		cmd       *cobra.Command
		args      []string
		toComplete string
		wantIDs   []string
		wantCount int
	}{
		{
			name:      "model download completion",
			cmd:       modelDownloadCmd,
			args:      []string{},
			toComplete: "",
			wantIDs:   []string{"qwen2.5-coder-3b-q4", "qwen2.5-coder-7b-q4", "qwen2.5-coder-7b-q5", "qwen2.5-coder-14b-q5"},
			wantCount: 4,
		},
		{
			name:      "model info completion",
			cmd:       modelInfoCmd,
			args:      []string{},
			toComplete: "",
			wantIDs:   []string{"qwen2.5-coder-3b-q4", "qwen2.5-coder-7b-q4"},
			wantCount: 4,
		},
		{
			name:      "model remove completion",
			cmd:       modelRemoveCmd,
			args:      []string{},
			toComplete: "",
			wantIDs:   []string{"qwen2.5-coder-3b-q4"},
			wantCount: 4,
		},
		{
			name:      "no completion after first arg",
			cmd:       modelDownloadCmd,
			args:      []string{"qwen2.5-coder-3b-q4"},
			toComplete: "",
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			completions, directive := tt.cmd.ValidArgsFunction(tt.cmd, tt.args, tt.toComplete)

			// Check count
			if len(completions) != tt.wantCount {
				t.Errorf("Got %d completions, want %d", len(completions), tt.wantCount)
			}

			// Check specific IDs if provided
			for _, wantID := range tt.wantIDs {
				found := false
				for _, completion := range completions {
					if strings.HasPrefix(completion, wantID) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected completion for model ID %q not found", wantID)
				}
			}

			// Check directive
			if tt.wantCount > 0 && directive != cobra.ShellCompDirectiveNoFileComp {
				t.Errorf("Expected NoFileComp directive, got %v", directive)
			}
		})
	}
}

func TestConfigKeyCompletions(t *testing.T) {
	// This test validates the completion function structure
	// The actual config command integration will be tested when config commands are implemented
	
	validConfigKeys := func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) != 0 {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}
		
		configKeys := []string{
			"default_model\tDefault model ID to use",
			"stream_mode\tEnable streaming mode by default",
			"temperature\tSampling temperature (0.0-2.0)",
		}
		
		return configKeys, cobra.ShellCompDirectiveNoFileComp
	}
	
	tests := []struct {
		name       string
		args       []string
		toComplete  string
		wantKeys   []string
		wantCount  int
	}{
		{
			name:       "config key completion",
			args:       []string{},
			toComplete:  "",
			wantKeys:   []string{"default_model", "stream_mode", "temperature"},
			wantCount:  3,
		},
		{
			name:       "no completion after first arg",
			args:       []string{"default_model"},
			toComplete:  "",
			wantCount:  0,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			completions, directive := validConfigKeys(nil, tt.args, tt.toComplete)
			
			// Check count
			if len(completions) != tt.wantCount {
				t.Errorf("Got %d completions, want %d", len(completions), tt.wantCount)
			}
			
			// Check specific keys if provided
			for _, wantKey := range tt.wantKeys {
				found := false
				for _, completion := range completions {
					if strings.HasPrefix(completion, wantKey) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected completion for config key %q not found", wantKey)
				}
			}
			
			// Check directive
			if directive != cobra.ShellCompDirectiveNoFileComp {
				t.Errorf("Expected NoFileComp directive, got %v", directive)
			}
		})
	}
}

func TestCompletionCommandHelp(t *testing.T) {
	cmd := createTestRootCommand()
	
	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	
	// Test help output
	cmd.SetArgs([]string{"completion", "--help"})
	
	err := cmd.Execute()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	output := buf.String()
	
	// Check that help contains key information
	expectedStrings := []string{
		"Generate shell completion script",
		"bash",
		"zsh",
		"fish",
		"powershell",
	}
	
	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("Help output missing expected string: %q", expected)
		}
	}
}

func TestCompletionValidArgs(t *testing.T) {
	// Test that only valid shells are accepted
	validShells := []string{"bash", "zsh", "fish", "powershell"}
	
	if len(completionCmd.ValidArgs) != len(validShells) {
		t.Errorf("Expected %d valid args, got %d", len(validShells), len(completionCmd.ValidArgs))
	}
	
	for _, shell := range validShells {
		found := false
		for _, validArg := range completionCmd.ValidArgs {
			if validArg == shell {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected shell %q not found in ValidArgs", shell)
		}
	}
}
