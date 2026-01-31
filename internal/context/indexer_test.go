package context

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewIndexer(t *testing.T) {
	tmpDir := t.TempDir()
	
	indexer, err := NewIndexer(tmpDir, DefaultIndexOptions())
	if err != nil {
		t.Fatalf("Failed to create indexer: %v", err)
	}
	
	if indexer.root != tmpDir {
		t.Errorf("Expected root %s, got %s", tmpDir, indexer.root)
	}
}

func TestIndexerWithGitignore(t *testing.T) {
	tmpDir := t.TempDir()
	
	// Create .gitignore
	gitignore := filepath.Join(tmpDir, ".gitignore")
	os.WriteFile(gitignore, []byte("*.log\nnode_modules/\n"), 0644)
	
	// Create test files
	os.WriteFile(filepath.Join(tmpDir, "main.go"), []byte("package main"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "test.log"), []byte("logs"), 0644)
	os.Mkdir(filepath.Join(tmpDir, "node_modules"), 0755)
	os.WriteFile(filepath.Join(tmpDir, "node_modules", "lib.js"), []byte("code"), 0644)
	
	indexer, _ := NewIndexer(tmpDir, DefaultIndexOptions())
	index, err := indexer.Index()
	if err != nil {
		t.Fatalf("Failed to index: %v", err)
	}
	
	// Should only include main.go
	if index.FileCount != 1 {
		t.Errorf("Expected 1 file, got %d", index.FileCount)
	}
	
	if _, ok := index.GetFile("main.go"); !ok {
		t.Error("Expected main.go to be indexed")
	}
	
	if _, ok := index.GetFile("test.log"); ok {
		t.Error("test.log should be ignored")
	}
}

func TestIndexFiles(t *testing.T) {
	tmpDir := t.TempDir()
	
	// Create test files
	files := []string{
		"main.go",
		"utils.go",
		"README.md",
		"test_test.go",
	}
	
	for _, file := range files {
		os.WriteFile(filepath.Join(tmpDir, file), []byte("content"), 0644)
	}
	
	indexer, _ := NewIndexer(tmpDir, DefaultIndexOptions())
	index, err := indexer.Index()
	if err != nil {
		t.Fatalf("Failed to index: %v", err)
	}
	
	if index.FileCount != len(files) {
		t.Errorf("Expected %d files, got %d", len(files), index.FileCount)
	}
	
	// Check file entries
	for _, file := range files {
		entry, ok := index.GetFile(file)
		if !ok {
			t.Errorf("File %s not indexed", file)
			continue
		}
		
		if entry.Path != file {
			t.Errorf("Expected path %s, got %s", file, entry.Path)
		}
		
		if entry.Size <= 0 {
			t.Errorf("File %s has invalid size: %d", file, entry.Size)
		}
	}
}

func TestDetectLanguage(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"main.go", "Go"},
		{"script.py", "Python"},
		{"index.js", "JavaScript"},
		{"app.ts", "TypeScript"},
		{"Main.java", "Java"},
		{"code.c", "C"},
		{"code.cpp", "C++"},
		{"lib.rs", "Rust"},
		{"README.md", "Markdown"},
		{"config.yaml", "YAML"},
		{"data.json", "JSON"},
		{"Makefile", "Configuration"},
		{"unknown.xyz", "Unknown"},
	}
	
	for _, tt := range tests {
		result := detectLanguage(tt.path)
		if result != tt.expected {
			t.Errorf("detectLanguage(%s) = %s; want %s", tt.path, result, tt.expected)
		}
	}
}

func TestIsTestFile(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"main_test.go", true},
		{"test_utils.py", true},
		{"app.test.js", true},
		{"component.spec.ts", true},
		{"main.go", false},
		{"utils.py", false},
		{"README.md", false},
	}
	
	for _, tt := range tests {
		result := isTestFile(tt.path)
		if result != tt.expected {
			t.Errorf("isTestFile(%s) = %v; want %v", tt.path, result, tt.expected)
		}
	}
}

func TestIsGeneratedFile(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"proto.pb.go", true},
		{"api.pb.gw.go", true},
		{"generated_code.go", true},
		{"gen_models.py", true},
		{"main.go", false},
		{"utils.py", false},
	}
	
	for _, tt := range tests {
		result := isGeneratedFile(tt.path)
		if result != tt.expected {
			t.Errorf("isGeneratedFile(%s) = %v; want %v", tt.path, result, tt.expected)
		}
	}
}

func TestGetFilesByLanguage(t *testing.T) {
	tmpDir := t.TempDir()
	
	// Create files in different languages
	files := map[string]string{
		"main.go":   "package main",
		"utils.go":  "package utils",
		"script.py": "print('hello')",
		"app.js":    "console.log('hi')",
	}
	
	for name, content := range files {
		os.WriteFile(filepath.Join(tmpDir, name), []byte(content), 0644)
	}
	
	indexer, _ := NewIndexer(tmpDir, DefaultIndexOptions())
	index, _ := indexer.Index()
	
	goFiles := index.GetFilesByLanguage("Go")
	if len(goFiles) != 2 {
		t.Errorf("Expected 2 Go files, got %d", len(goFiles))
	}
	
	pyFiles := index.GetFilesByLanguage("Python")
	if len(pyFiles) != 1 {
		t.Errorf("Expected 1 Python file, got %d", len(pyFiles))
	}
}

func TestGetFilesByPattern(t *testing.T) {
	tmpDir := t.TempDir()
	
	files := []string{"main.go", "utils.go", "README.md", "test_test.go"}
	for _, file := range files {
		os.WriteFile(filepath.Join(tmpDir, file), []byte("content"), 0644)
	}
	
	indexer, _ := NewIndexer(tmpDir, DefaultIndexOptions())
	index, _ := indexer.Index()
	
	goFiles := index.GetFilesByPattern("*.go")
	if len(goFiles) != 3 {
		t.Errorf("Expected 3 .go files, got %d", len(goFiles))
	}
	
	mdFiles := index.GetFilesByPattern("*.md")
	if len(mdFiles) != 1 {
		t.Errorf("Expected 1 .md file, got %d", len(mdFiles))
	}
}
