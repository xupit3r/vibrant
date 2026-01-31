package codeintel

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestNewCodeAnalyzer(t *testing.T) {
	ca := NewCodeAnalyzer()
	if ca == nil {
		t.Fatal("NewCodeAnalyzer returned nil")
	}
	
	if ca.symbols == nil || ca.references == nil || ca.imports == nil {
		t.Error("Analyzer maps not initialized")
	}
}

func TestAnalyzeGoFile(t *testing.T) {
	// Create temp directory with a test Go file
	tmpDir, err := os.MkdirTemp("", "vibrant-codeintel-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create a simple Go file
	testFile := filepath.Join(tmpDir, "test.go")
	content := `package testpkg

import "fmt"

type TestStruct struct {
	Name string
}

func TestFunction() error {
	return nil
}

func (t *TestStruct) TestMethod() {
	fmt.Println("test")
}

const TestConst = 42
var TestVar = "test"
`
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Analyze the file
	ca := NewCodeAnalyzer()
	err = ca.AnalyzeGoFile(context.Background(), testFile)
	if err != nil {
		t.Fatalf("AnalyzeGoFile failed: %v", err)
	}

	// Check symbols were extracted
	symbols := ca.GetSymbolsByPackage("testpkg")
	if len(symbols) == 0 {
		t.Fatal("No symbols extracted")
	}

	// Check for specific symbols
	foundFunc := false
	foundMethod := false
	foundStruct := false
	foundConst := false
	foundVar := false

	for _, sym := range symbols {
		switch sym.Name {
		case "TestFunction":
			foundFunc = true
			if sym.Type != "function" {
				t.Errorf("TestFunction has wrong type: %s", sym.Type)
			}
		case "TestMethod":
			foundMethod = true
			if sym.Type != "method" {
				t.Errorf("TestMethod has wrong type: %s", sym.Type)
			}
			if sym.Receiver != "*TestStruct" {
				t.Errorf("TestMethod has wrong receiver: %s", sym.Receiver)
			}
		case "TestStruct":
			foundStruct = true
			if sym.Type != "struct" {
				t.Errorf("TestStruct has wrong type: %s", sym.Type)
			}
		case "TestConst":
			foundConst = true
			if sym.Type != "const" {
				t.Errorf("TestConst has wrong type: %s", sym.Type)
			}
		case "TestVar":
			foundVar = true
			if sym.Type != "var" {
				t.Errorf("TestVar has wrong type: %s", sym.Type)
			}
		}
	}

	if !foundFunc {
		t.Error("TestFunction not found")
	}
	if !foundMethod {
		t.Error("TestMethod not found")
	}
	if !foundStruct {
		t.Error("TestStruct not found")
	}
	if !foundConst {
		t.Error("TestConst not found")
	}
	if !foundVar {
		t.Error("TestVar not found")
	}

	// Check imports
	imports := ca.GetImports(testFile)
	if len(imports) != 1 || imports[0] != "fmt" {
		t.Errorf("Expected imports [fmt], got %v", imports)
	}
}

func TestGetSymbolsByType(t *testing.T) {
	ca := NewCodeAnalyzer()
	ca.symbols["testpkg"] = []Symbol{
		{Name: "Func1", Type: "function"},
		{Name: "Func2", Type: "function"},
		{Name: "Type1", Type: "struct"},
	}

	functions := ca.GetSymbolsByType("function")
	if len(functions) != 2 {
		t.Errorf("Expected 2 functions, got %d", len(functions))
	}

	structs := ca.GetSymbolsByType("struct")
	if len(structs) != 1 {
		t.Errorf("Expected 1 struct, got %d", len(structs))
	}
}

func TestFindSymbol(t *testing.T) {
	ca := NewCodeAnalyzer()
	ca.symbols["pkg1"] = []Symbol{
		{Name: "Shared", Type: "function", Package: "pkg1"},
	}
	ca.symbols["pkg2"] = []Symbol{
		{Name: "Shared", Type: "function", Package: "pkg2"},
		{Name: "Unique", Type: "function", Package: "pkg2"},
	}

	shared := ca.FindSymbol("Shared")
	if len(shared) != 2 {
		t.Errorf("Expected 2 'Shared' symbols, got %d", len(shared))
	}

	unique := ca.FindSymbol("Unique")
	if len(unique) != 1 {
		t.Errorf("Expected 1 'Unique' symbol, got %d", len(unique))
	}

	notfound := ca.FindSymbol("NotExist")
	if len(notfound) != 0 {
		t.Errorf("Expected 0 symbols for 'NotExist', got %d", len(notfound))
	}
}

func TestGetDependencies(t *testing.T) {
	ca := NewCodeAnalyzer()
	ca.imports["file1.go"] = []string{"fmt", "os"}
	ca.imports["file2.go"] = []string{"fmt", "context"}

	deps := ca.GetDependencies()
	if len(deps) != 3 {
		t.Errorf("Expected 3 unique dependencies, got %d", len(deps))
	}

	// Check that all deps are present
	depsMap := make(map[string]bool)
	for _, dep := range deps {
		depsMap[dep] = true
	}

	if !depsMap["fmt"] || !depsMap["os"] || !depsMap["context"] {
		t.Errorf("Missing expected dependencies, got: %v", deps)
	}
}

func TestGetPackages(t *testing.T) {
	ca := NewCodeAnalyzer()
	ca.symbols["pkg1"] = []Symbol{{Name: "Test"}}
	ca.symbols["pkg2"] = []Symbol{{Name: "Test"}}

	pkgs := ca.GetPackages()
	if len(pkgs) != 2 {
		t.Errorf("Expected 2 packages, got %d", len(pkgs))
	}
}

func TestGetSymbolCount(t *testing.T) {
	ca := NewCodeAnalyzer()
	ca.symbols["pkg1"] = []Symbol{{Name: "Test1"}, {Name: "Test2"}}
	ca.symbols["pkg2"] = []Symbol{{Name: "Test3"}}

	count := ca.GetSymbolCount()
	if count != 3 {
		t.Errorf("Expected 3 symbols, got %d", count)
	}
}

func TestAnalyzeGoDirectory(t *testing.T) {
	// Create temp directory structure
	tmpDir, err := os.MkdirTemp("", "vibrant-codeintel-dir-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create subdirectory
	subDir := filepath.Join(tmpDir, "subpkg")
	if err := os.Mkdir(subDir, 0755); err != nil {
		t.Fatalf("Failed to create subdir: %v", err)
	}

	// Create Go files
	file1 := filepath.Join(tmpDir, "file1.go")
	file2 := filepath.Join(subDir, "file2.go")

	content1 := `package main
func Main() {}`
	content2 := `package subpkg
func Sub() {}`

	if err := os.WriteFile(file1, []byte(content1), 0644); err != nil {
		t.Fatalf("Failed to write file1: %v", err)
	}
	if err := os.WriteFile(file2, []byte(content2), 0644); err != nil {
		t.Fatalf("Failed to write file2: %v", err)
	}

	// Analyze directory
	ca := NewCodeAnalyzer()
	err = ca.AnalyzeGoDirectory(context.Background(), tmpDir)
	if err != nil {
		t.Fatalf("AnalyzeGoDirectory failed: %v", err)
	}

	// Should find symbols from both packages
	pkgs := ca.GetPackages()
	if len(pkgs) < 1 {
		t.Errorf("Expected at least 1 package, got %d", len(pkgs))
	}

	count := ca.GetSymbolCount()
	if count < 2 {
		t.Errorf("Expected at least 2 symbols, got %d", count)
	}
}
