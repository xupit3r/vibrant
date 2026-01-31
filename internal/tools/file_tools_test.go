package tools

import (
"context"
"os"
"path/filepath"
"testing"
)

func TestReadFileTool(t *testing.T) {
tmpDir := t.TempDir()
testFile := filepath.Join(tmpDir, "test.txt")
testContent := "Hello, World!"

if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
t.Fatalf("Failed to create test file: %v", err)
}

tool := &ReadFileTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path": testFile,
})

if err != nil {
t.Fatalf("Execute failed: %v", err)
}

if !result.Success {
t.Error("Expected successful execution")
}

if result.Output != testContent {
t.Errorf("Expected output '%s', got '%s'", testContent, result.Output)
}
}

func TestReadFileToolNonExistent(t *testing.T) {
tool := &ReadFileTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path": "/nonexistent/file.txt",
})

if err == nil {
t.Error("Expected error for non-existent file")
}

if result.Success {
t.Error("Result should indicate failure")
}
}

func TestWriteFileTool(t *testing.T) {
tmpDir := t.TempDir()
testFile := filepath.Join(tmpDir, "output.txt")
testContent := "Test content"

tool := &WriteFileTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path":    testFile,
"content": testContent,
})

if err != nil {
t.Fatalf("Execute failed: %v", err)
}

if !result.Success {
t.Error("Expected successful execution")
}

// Verify file was written
content, err := os.ReadFile(testFile)
if err != nil {
t.Fatalf("Failed to read written file: %v", err)
}

if string(content) != testContent {
t.Errorf("File content mismatch: expected '%s', got '%s'", testContent, string(content))
}
}

func TestWriteFileToolCreatesDirectory(t *testing.T) {
tmpDir := t.TempDir()
testFile := filepath.Join(tmpDir, "subdir", "nested", "file.txt")
testContent := "Nested content"

tool := &WriteFileTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path":    testFile,
"content": testContent,
})

if err != nil {
t.Fatalf("Execute failed: %v", err)
}

if !result.Success {
t.Error("Expected successful execution")
}

// Verify file was written
if _, err := os.Stat(testFile); os.IsNotExist(err) {
t.Error("File should have been created")
}
}

func TestListDirectoryTool(t *testing.T) {
tmpDir := t.TempDir()

// Create test files
os.WriteFile(filepath.Join(tmpDir, "file1.txt"), []byte("test"), 0644)
os.WriteFile(filepath.Join(tmpDir, "file2.txt"), []byte("test"), 0644)
os.Mkdir(filepath.Join(tmpDir, "subdir"), 0755)

tool := &ListDirectoryTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path": tmpDir,
})

if err != nil {
t.Fatalf("Execute failed: %v", err)
}

if !result.Success {
t.Error("Expected successful execution")
}

// Check data contains expected entries
data, ok := result.Data.(map[string]interface{})
if !ok {
t.Fatal("Result data should be a map")
}

count, ok := data["count"].(int)
if !ok || count != 3 {
t.Errorf("Expected 3 entries, got %v", data["count"])
}
}

func TestListDirectoryToolNonExistent(t *testing.T) {
tool := &ListDirectoryTool{}
ctx := context.Background()

result, err := tool.Execute(ctx, map[string]interface{}{
"path": "/nonexistent/directory",
})

if err == nil {
t.Error("Expected error for non-existent directory")
}

if result.Success {
t.Error("Result should indicate failure")
}
}

func TestToolParameters(t *testing.T) {
tests := []struct {
name string
tool Tool
}{
{"ReadFileTool", &ReadFileTool{}},
{"WriteFileTool", &WriteFileTool{}},
{"ListDirectoryTool", &ListDirectoryTool{}},
}

for _, tt := range tests {
t.Run(tt.name, func(t *testing.T) {
if tt.tool.Name() == "" {
t.Error("Tool name should not be empty")
}

if tt.tool.Description() == "" {
t.Error("Tool description should not be empty")
}

params := tt.tool.Parameters()
if len(params) == 0 {
t.Error("Tool should have parameters")
}
})
}
}
