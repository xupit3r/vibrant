package tools

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// ShellTool executes shell commands
type ShellTool struct {
	timeout time.Duration
}

// NewShellTool creates a new shell command execution tool
func NewShellTool() *ShellTool {
	return &ShellTool{
		timeout: 30 * time.Second, // Default timeout
	}
}

// SetTimeout sets the execution timeout
func (s *ShellTool) SetTimeout(timeout time.Duration) {
	s.timeout = timeout
}

func (s *ShellTool) Name() string {
	return "shell"
}

func (s *ShellTool) Description() string {
	return "Execute shell commands with timeout protection"
}

func (s *ShellTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"command": {
			Name:        "command",
			Description: "Shell command to execute",
			Type:        "string",
			Required:    true,
		},
		"workdir": {
			Name:        "workdir",
			Description: "Working directory (optional, defaults to current directory)",
			Type:        "string",
			Required:    false,
		},
	}
}

func (s *ShellTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	command, ok := params["command"].(string)
	if !ok || command == "" {
		return &Result{
			Success: false,
			Error:   "command parameter is required and must be a string",
		}, nil
	}

	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	// Create the command
	cmd := exec.CommandContext(timeoutCtx, "sh", "-c", command)

	// Set working directory if provided
	if workdir, ok := params["workdir"].(string); ok && workdir != "" {
		cmd.Dir = workdir
	}

	// Execute command
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Check if it was a timeout
		if timeoutCtx.Err() == context.DeadlineExceeded {
			return &Result{
				Success: false,
				Output:  string(output),
				Error:   fmt.Sprintf("command timed out after %v", s.timeout),
			}, nil
		}

		return &Result{
			Success: false,
			Output:  string(output),
			Error:   fmt.Sprintf("command failed: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  strings.TrimSpace(string(output)),
	}, nil
}

// GrepTool searches for patterns in files
type GrepTool struct{}

func (g *GrepTool) Name() string {
	return "grep"
}

func (g *GrepTool) Description() string {
	return "Search for patterns in files using grep"
}

func (g *GrepTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"pattern": {
			Name:        "pattern",
			Description: "Search pattern (regex)",
			Type:        "string",
			Required:    true,
		},
		"path": {
			Name:        "path",
			Description: "File or directory to search",
			Type:        "string",
			Required:    true,
		},
		"recursive": {
			Name:        "recursive",
			Description: "Search recursively in directories",
			Type:        "bool",
			Required:    false,
		},
	}
}

func (g *GrepTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	pattern, ok := params["pattern"].(string)
	if !ok || pattern == "" {
		return &Result{
			Success: false,
			Error:   "pattern parameter is required",
		}, nil
	}

	path, ok := params["path"].(string)
	if !ok || path == "" {
		return &Result{
			Success: false,
			Error:   "path parameter is required",
		}, nil
	}

	args := []string{"-n", pattern, path}
	
	// Add recursive flag if specified
	if recursive, ok := params["recursive"].(bool); ok && recursive {
		args = []string{"-rn", pattern, path}
	}

	cmd := exec.CommandContext(ctx, "grep", args...)
	output, err := cmd.CombinedOutput()

	// grep returns exit code 1 when no matches found
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			return &Result{
				Success: true,
				Output:  "No matches found",
			}, nil
		}
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("grep failed: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  string(output),
	}, nil
}

// FindFilesTool finds files by name pattern
type FindFilesTool struct{}

func (f *FindFilesTool) Name() string {
	return "find_files"
}

func (f *FindFilesTool) Description() string {
	return "Find files by name pattern"
}

func (f *FindFilesTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"pattern": {
			Name:        "pattern",
			Description: "File name pattern (supports wildcards)",
			Type:        "string",
			Required:    true,
		},
		"path": {
			Name:        "path",
			Description: "Directory to search (defaults to current directory)",
			Type:        "string",
			Required:    false,
		},
	}
}

func (f *FindFilesTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	pattern, ok := params["pattern"].(string)
	if !ok || pattern == "" {
		return &Result{
			Success: false,
			Error:   "pattern parameter is required",
		}, nil
	}

	searchPath := "."
	if path, ok := params["path"].(string); ok && path != "" {
		searchPath = path
	}

	cmd := exec.CommandContext(ctx, "find", searchPath, "-name", pattern, "-type", "f")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("find failed: %v", err),
		}, nil
	}

	result := strings.TrimSpace(string(output))
	if result == "" {
		result = "No files found"
	}

	return &Result{
		Success: true,
		Output:  result,
	}, nil
}

// GetFileInfoTool gets file metadata
type GetFileInfoTool struct{}

func (g *GetFileInfoTool) Name() string {
	return "get_file_info"
}

func (g *GetFileInfoTool) Description() string {
	return "Get file metadata (size, permissions, modification time)"
}

func (g *GetFileInfoTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "Path to file",
			Type:        "string",
			Required:    true,
		},
	}
}

func (g *GetFileInfoTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, ok := params["path"].(string)
	if !ok || path == "" {
		return &Result{
			Success: false,
			Error:   "path parameter is required",
		}, nil
	}

	cmd := exec.CommandContext(ctx, "ls", "-lh", path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to get file info: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  strings.TrimSpace(string(output)),
	}, nil
}
