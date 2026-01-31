package integration

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"
)

// TestShellCompletionGeneration tests that completion scripts can be generated
func TestShellCompletionGeneration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	tests := []struct {
		name     string
		shell    string
		contains []string
	}{
		{
			name:  "bash completion generation",
			shell: "bash",
			contains: []string{
				"vibrant",
				"__start_vibrant",
				"__vibrant_handle_command",
			},
		},
		{
			name:  "zsh completion generation",
			shell: "zsh",
			contains: []string{
				"#compdef vibrant",
				"_vibrant",
			},
		},
		{
			name:  "fish completion generation",
			shell: "fish",
			contains: []string{
				"vibrant",
				"complete",
			},
		},
		{
			name:  "powershell completion generation",
			shell: "powershell",
			contains: []string{
				"vibrant",
				"Register-ArgumentCompleter",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("./vibrant", "completion", tt.shell)
			
			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr

			err := cmd.Run()
			if err != nil {
				t.Fatalf("Command failed: %v\nStderr: %s", err, stderr.String())
			}

			output := stdout.String()

			// Check that output contains expected strings
			for _, expected := range tt.contains {
				if !strings.Contains(output, expected) {
					t.Errorf("Output missing expected string: %q\nGot first 500 chars: %s",
						expected, truncate(output, 500))
				}
			}

			// Check that output is not empty
			if len(output) == 0 {
				t.Error("Completion output is empty")
			}
		})
	}
}

// TestCompletionSubcommands verifies completion includes all subcommands
func TestCompletionSubcommands(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	cmd := exec.Command("./vibrant", "completion", "bash")
	
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	err := cmd.Run()
	if err != nil {
		t.Fatalf("Command failed: %v", err)
	}

	output := stdout.String()

	// Verify that major subcommands are included in completion
	expectedCommands := []string{
		"ask",
		"chat",
		"model",
		"version",
		"completion",
	}

	for _, cmdName := range expectedCommands {
		if !strings.Contains(output, cmdName) {
			t.Errorf("Completion script missing command: %q", cmdName)
		}
	}
}

// TestCompletionModelSubcommands verifies model subcommands are in completion
func TestCompletionModelSubcommands(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	cmd := exec.Command("./vibrant", "completion", "bash")
	
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	err := cmd.Run()
	if err != nil {
		t.Fatalf("Command failed: %v", err)
	}

	output := stdout.String()

	// Verify model subcommands
	expectedSubcommands := []string{
		"list",
		"download",
		"info",
		"remove",
	}

	for _, subCmd := range expectedSubcommands {
		if !strings.Contains(output, subCmd) {
			t.Errorf("Completion script missing model subcommand: %q", subCmd)
		}
	}
}

// TestCompletionFlags verifies common flags are in completion
func TestCompletionFlags(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	cmd := exec.Command("./vibrant", "completion", "bash")
	
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	err := cmd.Run()
	if err != nil {
		t.Fatalf("Command failed: %v", err)
	}

	output := stdout.String()

	// Verify common flags
	expectedFlags := []string{
		"--help",
		"--verbose",
		"--quiet",
		"--config",
		"--no-color",
	}

	for _, flag := range expectedFlags {
		if !strings.Contains(output, flag) {
			t.Errorf("Completion script missing flag: %q", flag)
		}
	}
}

// TestCompletionErrorHandling verifies error handling for invalid shells
func TestCompletionErrorHandling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{
			name:    "invalid shell",
			args:    []string{"completion", "invalid"},
			wantErr: true,
		},
		{
			name:    "no arguments",
			args:    []string{"completion"},
			wantErr: true,
		},
		{
			name:    "too many arguments",
			args:    []string{"completion", "bash", "extra"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("./vibrant", tt.args...)
			
			var stderr bytes.Buffer
			cmd.Stderr = &stderr

			err := cmd.Run()
			
			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			
			if !tt.wantErr && err != nil {
				t.Errorf("Unexpected error: %v\nStderr: %s", err, stderr.String())
			}
		})
	}
}

// TestCompletionHelpText verifies help is available
func TestCompletionHelpText(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	cmd := exec.Command("./vibrant", "completion", "--help")
	
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	err := cmd.Run()
	if err != nil {
		t.Fatalf("Command failed: %v", err)
	}

	output := stdout.String()

	expectedStrings := []string{
		"Generate shell completion script",
		"bash",
		"zsh",
		"fish",
		"powershell",
		"Usage:",
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("Help output missing expected string: %q", expected)
		}
	}
}

// Helper function to truncate strings for error messages
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
