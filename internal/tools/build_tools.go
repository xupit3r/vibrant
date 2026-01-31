package tools

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// RunTestsTool executes tests and parses results
type RunTestsTool struct{}

func (r *RunTestsTool) Name() string {
	return "run_tests"
}

func (r *RunTestsTool) Description() string {
	return "Run tests and parse results (supports Go, Python, Node.js)"
}

func (r *RunTestsTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "Path to test file or directory",
			Type:        "string",
			Required:    false,
		},
		"language": {
			Name:        "language",
			Description: "Language: go, python, node (auto-detected if not specified)",
			Type:        "string",
			Required:    false,
		},
		"verbose": {
			Name:        "verbose",
			Description: "Verbose output",
			Type:        "bool",
			Required:    false,
		},
	}
}

func (r *RunTestsTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, _ := params["path"].(string)
	if path == "" {
		path = "."
	}

	language, _ := params["language"].(string)
	verbose, _ := params["verbose"].(bool)

	// Auto-detect language if not specified
	if language == "" {
		language = r.detectLanguage(path)
	}

	// Build test command
	var cmd *exec.Cmd
	switch strings.ToLower(language) {
	case "go", "golang":
		args := []string{"test"}
		if verbose {
			args = append(args, "-v")
		}
		args = append(args, path)
		cmd = exec.CommandContext(ctx, "go", args...)
	case "python", "py":
		args := []string{"-m", "pytest"}
		if verbose {
			args = append(args, "-v")
		}
		args = append(args, path)
		cmd = exec.CommandContext(ctx, "python3", args...)
	case "node", "javascript", "js":
		cmd = exec.CommandContext(ctx, "npm", "test")
	default:
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("unsupported language: %s", language),
		}, nil
	}

	// Run tests
	output, err := cmd.CombinedOutput()
	outputStr := strings.TrimSpace(string(output))

	if err != nil {
		// Tests may have failed
		return &Result{
			Success: false,
			Output:  outputStr,
			Error:   "tests failed",
		}, nil
	}

	return &Result{
		Success: true,
		Output:  outputStr,
	}, nil
}

func (r *RunTestsTool) detectLanguage(path string) string {
	// Check for Go files
	cmd := exec.Command("find", path, "-name", "*.go", "-type", "f")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "go"
	}

	// Check for Python files
	cmd = exec.Command("find", path, "-name", "*.py", "-type", "f")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "python"
	}

	// Check for Node.js
	cmd = exec.Command("find", path, "-name", "package.json", "-type", "f")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "node"
	}

	return "unknown"
}

// BuildTool builds projects
type BuildTool struct{}

func (b *BuildTool) Name() string {
	return "build"
}

func (b *BuildTool) Description() string {
	return "Build project (supports Go, Python, Node.js, Make)"
}

func (b *BuildTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "Project directory path",
			Type:        "string",
			Required:    false,
		},
		"tool": {
			Name:        "tool",
			Description: "Build tool: go, make, npm, pip (auto-detected if not specified)",
			Type:        "string",
			Required:    false,
		},
	}
}

func (b *BuildTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, _ := params["path"].(string)
	if path == "" {
		path = "."
	}

	tool, _ := params["tool"].(string)
	if tool == "" {
		tool = b.detectBuildTool(path)
	}

	// Build command
	var cmd *exec.Cmd
	switch strings.ToLower(tool) {
	case "go", "golang":
		cmd = exec.CommandContext(ctx, "go", "build", "./...")
		cmd.Dir = path
	case "make", "makefile":
		cmd = exec.CommandContext(ctx, "make")
		cmd.Dir = path
	case "npm", "node":
		cmd = exec.CommandContext(ctx, "npm", "run", "build")
		cmd.Dir = path
	case "pip", "python":
		cmd = exec.CommandContext(ctx, "pip", "install", "-e", ".")
		cmd.Dir = path
	default:
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("unsupported build tool: %s", tool),
		}, nil
	}

	// Run build
	output, err := cmd.CombinedOutput()
	outputStr := strings.TrimSpace(string(output))

	if err != nil {
		return &Result{
			Success: false,
			Output:  outputStr,
			Error:   fmt.Sprintf("build failed: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  outputStr,
	}, nil
}

func (b *BuildTool) detectBuildTool(path string) string {
	// Check for Go module
	cmd := exec.Command("find", path, "-maxdepth", "1", "-name", "go.mod")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "go"
	}

	// Check for Makefile
	cmd = exec.Command("find", path, "-maxdepth", "1", "-name", "Makefile")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "make"
	}

	// Check for package.json
	cmd = exec.Command("find", path, "-maxdepth", "1", "-name", "package.json")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "npm"
	}

	// Check for setup.py or pyproject.toml
	cmd = exec.Command("find", path, "-maxdepth", "1", "-name", "setup.py")
	if output, err := cmd.Output(); err == nil && len(output) > 0 {
		return "pip"
	}

	return "unknown"
}

// LintTool runs linters
type LintTool struct{}

func (l *LintTool) Name() string {
	return "lint"
}

func (l *LintTool) Description() string {
	return "Run linters (golangci-lint, pylint, eslint)"
}

func (l *LintTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "Path to lint",
			Type:        "string",
			Required:    false,
		},
		"linter": {
			Name:        "linter",
			Description: "Linter: golangci-lint, pylint, eslint (auto-detected if not specified)",
			Type:        "string",
			Required:    false,
		},
	}
}

func (l *LintTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, _ := params["path"].(string)
	if path == "" {
		path = "./..."
	}

	linter, _ := params["linter"].(string)
	if linter == "" {
		linter = l.detectLinter()
	}

	// Build lint command
	var cmd *exec.Cmd
	switch strings.ToLower(linter) {
	case "golangci-lint", "go":
		cmd = exec.CommandContext(ctx, "golangci-lint", "run", path)
	case "pylint", "python":
		cmd = exec.CommandContext(ctx, "pylint", path)
	case "eslint", "javascript", "js":
		cmd = exec.CommandContext(ctx, "eslint", path)
	default:
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("unsupported linter: %s", linter),
		}, nil
	}

	// Run linter
	output, err := cmd.CombinedOutput()
	outputStr := strings.TrimSpace(string(output))

	// Linters often return non-zero exit codes for issues found
	if err != nil && outputStr != "" {
		return &Result{
			Success: true, // Not a failure - just found issues
			Output:  outputStr,
		}, nil
	}

	if outputStr == "" {
		outputStr = "No issues found"
	}

	return &Result{
		Success: true,
		Output:  outputStr,
	}, nil
}

func (l *LintTool) detectLinter() string {
	// Check for golangci-lint
	if _, err := exec.LookPath("golangci-lint"); err == nil {
		return "golangci-lint"
	}

	// Check for pylint
	if _, err := exec.LookPath("pylint"); err == nil {
		return "pylint"
	}

	// Check for eslint
	if _, err := exec.LookPath("eslint"); err == nil {
		return "eslint"
	}

	return "unknown"
}
