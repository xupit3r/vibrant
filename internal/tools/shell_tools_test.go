package tools

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestShellTool(t *testing.T) {
	tool := NewShellTool()

	tests := []struct {
		name      string
		params    map[string]interface{}
		expectErr bool
		contains  string
	}{
		{
			name:      "simple echo",
			params:    map[string]interface{}{"command": "echo hello"},
			expectErr: false,
			contains:  "hello",
		},
		{
			name:      "list files",
			params:    map[string]interface{}{"command": "ls -la"},
			expectErr: false,
			contains:  "",
		},
		{
			name:      "missing command",
			params:    map[string]interface{}{},
			expectErr: true,
			contains:  "",
		},
		{
			name:      "invalid command",
			params:    map[string]interface{}{"command": "nonexistentcommand123"},
			expectErr: true,
			contains:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, _ := tool.Execute(context.Background(), tt.params)
			
			if tt.expectErr && result.Success {
				t.Error("Expected command to fail")
			}
			
			if !tt.expectErr && !result.Success {
				t.Errorf("Expected command to succeed, got error: %s", result.Error)
			}

			if tt.contains != "" && !contains(result.Output, tt.contains) {
				t.Errorf("Expected output to contain '%s', got '%s'", tt.contains, result.Output)
			}
		})
	}
}

func TestShellToolTimeout(t *testing.T) {
	tool := NewShellTool()
	tool.SetTimeout(100 * time.Millisecond)

	params := map[string]interface{}{
		"command": "sleep 1",
	}

	result, _ := tool.Execute(context.Background(), params)
	if result.Success {
		t.Error("Expected command to timeout")
	}

	if !contains(result.Error, "timed out") {
		t.Errorf("Expected timeout error, got: %s", result.Error)
	}
}

func TestShellToolWorkdir(t *testing.T) {
	tool := NewShellTool()
	
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "vibrant-shell-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	params := map[string]interface{}{
		"command": "pwd",
		"workdir": tmpDir,
	}

	result, _ := tool.Execute(context.Background(), params)
	if !result.Success {
		t.Errorf("Expected command to succeed: %s", result.Error)
	}

	if !contains(result.Output, tmpDir) {
		t.Errorf("Expected output to contain '%s', got '%s'", tmpDir, result.Output)
	}
}

func TestGrepTool(t *testing.T) {
	tool := &GrepTool{}

	// Create temp file with content
	tmpDir, err := os.MkdirTemp("", "vibrant-grep-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test.txt")
	content := "hello world\nfoo bar\nhello again"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	tests := []struct {
		name      string
		params    map[string]interface{}
		expectErr bool
		contains  string
	}{
		{
			name: "find pattern",
			params: map[string]interface{}{
				"pattern": "hello",
				"path":    testFile,
			},
			expectErr: false,
			contains:  "hello",
		},
		{
			name: "no matches",
			params: map[string]interface{}{
				"pattern": "notfound",
				"path":    testFile,
			},
			expectErr: false,
			contains:  "No matches",
		},
		{
			name: "missing pattern",
			params: map[string]interface{}{
				"path": testFile,
			},
			expectErr: true,
			contains:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, _ := tool.Execute(context.Background(), tt.params)
			
			if tt.expectErr && result.Success {
				t.Error("Expected tool to fail")
			}
			
			if !tt.expectErr && !result.Success {
				t.Errorf("Expected tool to succeed, got error: %s", result.Error)
			}

			if tt.contains != "" && !contains(result.Output, tt.contains) {
				t.Errorf("Expected output to contain '%s', got '%s'", tt.contains, result.Output)
			}
		})
	}
}

func TestGrepToolRecursive(t *testing.T) {
	tool := &GrepTool{}

	// Create temp directory structure
	tmpDir, err := os.MkdirTemp("", "vibrant-grep-recursive-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create subdirectory
	subDir := filepath.Join(tmpDir, "subdir")
	if err := os.Mkdir(subDir, 0755); err != nil {
		t.Fatalf("Failed to create subdir: %v", err)
	}

	// Create files
	file1 := filepath.Join(tmpDir, "file1.txt")
	file2 := filepath.Join(subDir, "file2.txt")
	
	if err := os.WriteFile(file1, []byte("test pattern"), 0644); err != nil {
		t.Fatalf("Failed to write file1: %v", err)
	}
	if err := os.WriteFile(file2, []byte("test pattern in subdir"), 0644); err != nil {
		t.Fatalf("Failed to write file2: %v", err)
	}

	params := map[string]interface{}{
		"pattern":   "test pattern",
		"path":      tmpDir,
		"recursive": true,
	}

	result, _ := tool.Execute(context.Background(), params)
	if !result.Success {
		t.Errorf("Expected recursive grep to succeed: %s", result.Error)
	}

	// Should find matches in both files
	if !contains(result.Output, "file1.txt") || !contains(result.Output, "file2.txt") {
		t.Errorf("Expected to find matches in both files, got: %s", result.Output)
	}
}

func TestFindFilesTool(t *testing.T) {
	tool := &FindFilesTool{}

	// Create temp directory with test files
	tmpDir, err := os.MkdirTemp("", "vibrant-find-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create test files
	testFiles := []string{"test.go", "test.txt", "foo.go"}
	for _, name := range testFiles {
		file := filepath.Join(tmpDir, name)
		if err := os.WriteFile(file, []byte("test"), 0644); err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
	}

	tests := []struct {
		name      string
		params    map[string]interface{}
		expectErr bool
		contains  string
	}{
		{
			name: "find go files",
			params: map[string]interface{}{
				"pattern": "*.go",
				"path":    tmpDir,
			},
			expectErr: false,
			contains:  "test.go",
		},
		{
			name: "find txt files",
			params: map[string]interface{}{
				"pattern": "*.txt",
				"path":    tmpDir,
			},
			expectErr: false,
			contains:  "test.txt",
		},
		{
			name: "no matches",
			params: map[string]interface{}{
				"pattern": "*.xyz",
				"path":    tmpDir,
			},
			expectErr: false,
			contains:  "No files found",
		},
		{
			name: "missing pattern",
			params: map[string]interface{}{
				"path": tmpDir,
			},
			expectErr: true,
			contains:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, _ := tool.Execute(context.Background(), tt.params)
			
			if tt.expectErr && result.Success {
				t.Error("Expected tool to fail")
			}
			
			if !tt.expectErr && !result.Success {
				t.Errorf("Expected tool to succeed, got error: %s", result.Error)
			}

			if tt.contains != "" && !contains(result.Output, tt.contains) {
				t.Errorf("Expected output to contain '%s', got '%s'", tt.contains, result.Output)
			}
		})
	}
}

func TestGetFileInfoTool(t *testing.T) {
	tool := &GetFileInfoTool{}

	// Create temp file
	tmpFile, err := os.CreateTemp("", "vibrant-fileinfo-test")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString("test content"); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	tests := []struct {
		name      string
		params    map[string]interface{}
		expectErr bool
	}{
		{
			name: "get file info",
			params: map[string]interface{}{
				"path": tmpFile.Name(),
			},
			expectErr: false,
		},
		{
			name: "nonexistent file",
			params: map[string]interface{}{
				"path": "/nonexistent/file.txt",
			},
			expectErr: true,
		},
		{
			name:      "missing path",
			params:    map[string]interface{}{},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, _ := tool.Execute(context.Background(), tt.params)
			
			if tt.expectErr && result.Success {
				t.Error("Expected tool to fail")
			}
			
			if !tt.expectErr && !result.Success {
				t.Errorf("Expected tool to succeed, got error: %s", result.Error)
			}
		})
	}
}

func TestShellToolParameters(t *testing.T) {
	tests := []struct {
		name     string
		tool     Tool
		minParams int
	}{
		{"ShellTool", NewShellTool(), 1},
		{"GrepTool", &GrepTool{}, 2},
		{"FindFilesTool", &FindFilesTool{}, 1},
		{"GetFileInfoTool", &GetFileInfoTool{}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := tt.tool.Parameters()
			if len(params) < tt.minParams {
				t.Errorf("Expected at least %d parameters, got %d", tt.minParams, len(params))
			}

			// Check that each parameter has required fields
			for _, param := range params {
				if param.Name == "" {
					t.Error("Parameter missing name")
				}
				if param.Description == "" {
					t.Error("Parameter missing description")
				}
				if param.Type == "" {
					t.Error("Parameter missing type")
				}
			}
		})
	}
}
