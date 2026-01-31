package codeintel

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// Symbol represents a code symbol (function, type, variable, etc.)
type Symbol struct {
	Name     string
	Type     string // "function", "type", "const", "var", "struct", "interface", "method"
	Package  string
	File     string
	Line     int
	Receiver string // For methods
	Signature string // Function/method signature
}

// Reference represents a usage of a symbol
type Reference struct {
	Symbol Symbol
	File   string
	Line   int
	Column int
	Kind   string // "definition", "call", "declaration", "import"
}

// CodeAnalyzer analyzes code structure
type CodeAnalyzer struct {
	symbols    map[string][]Symbol // package -> symbols
	references map[string][]Reference // symbol name -> references
	imports    map[string][]string // file -> imported packages
}

// NewCodeAnalyzer creates a new code analyzer
func NewCodeAnalyzer() *CodeAnalyzer {
	return &CodeAnalyzer{
		symbols:    make(map[string][]Symbol),
		references: make(map[string][]Reference),
		imports:    make(map[string][]string),
	}
}

// AnalyzeGoFile analyzes a single Go source file
func (ca *CodeAnalyzer) AnalyzeGoFile(ctx context.Context, path string) error {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Read file content
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Parse the file
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, content, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse file: %w", err)
	}

	// Extract package name
	pkgName := file.Name.Name

	// Extract imports
	imports := make([]string, 0)
	for _, imp := range file.Imports {
		importPath := strings.Trim(imp.Path.Value, `"`)
		imports = append(imports, importPath)
	}
	ca.imports[path] = imports

	// Extract symbols
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			symbol := Symbol{
				Name:    node.Name.Name,
				Type:    "function",
				Package: pkgName,
				File:    path,
				Line:    fset.Position(node.Pos()).Line,
			}

			// Check if it's a method
			if node.Recv != nil && len(node.Recv.List) > 0 {
				symbol.Type = "method"
				// Extract receiver type
				if starExpr, ok := node.Recv.List[0].Type.(*ast.StarExpr); ok {
					if ident, ok := starExpr.X.(*ast.Ident); ok {
						symbol.Receiver = "*" + ident.Name
					}
				} else if ident, ok := node.Recv.List[0].Type.(*ast.Ident); ok {
					symbol.Receiver = ident.Name
				}
			}

			// Build signature
			signature := fmt.Sprintf("func %s(", symbol.Name)
			if node.Type.Params != nil {
				params := make([]string, 0)
				for _, field := range node.Type.Params.List {
					typeStr := exprToString(field.Type)
					for range field.Names {
						params = append(params, typeStr)
					}
				}
				signature += strings.Join(params, ", ")
			}
			signature += ")"
			
			if node.Type.Results != nil && len(node.Type.Results.List) > 0 {
				results := make([]string, 0)
				for _, field := range node.Type.Results.List {
					results = append(results, exprToString(field.Type))
				}
				if len(results) == 1 {
					signature += " " + results[0]
				} else {
					signature += " (" + strings.Join(results, ", ") + ")"
				}
			}
			symbol.Signature = signature

			ca.symbols[pkgName] = append(ca.symbols[pkgName], symbol)

		case *ast.GenDecl:
			// Handle type, const, var declarations
			for _, spec := range node.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					symbol := Symbol{
						Name:    s.Name.Name,
						Package: pkgName,
						File:    path,
						Line:    fset.Position(s.Pos()).Line,
					}
					
					// Determine type
					switch s.Type.(type) {
					case *ast.StructType:
						symbol.Type = "struct"
					case *ast.InterfaceType:
						symbol.Type = "interface"
					default:
						symbol.Type = "type"
					}

					ca.symbols[pkgName] = append(ca.symbols[pkgName], symbol)

				case *ast.ValueSpec:
					varType := "var"
					if node.Tok == token.CONST {
						varType = "const"
					}
					
					for _, name := range s.Names {
						symbol := Symbol{
							Name:    name.Name,
							Type:    varType,
							Package: pkgName,
							File:    path,
							Line:    fset.Position(name.Pos()).Line,
						}
						ca.symbols[pkgName] = append(ca.symbols[pkgName], symbol)
					}
				}
			}
		}
		return true
	})

	return nil
}

// AnalyzeGoDirectory analyzes all Go files in a directory recursively
func (ca *CodeAnalyzer) AnalyzeGoDirectory(ctx context.Context, dir string) error {
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip hidden directories and vendor
		if info.IsDir() {
			name := filepath.Base(path)
			if strings.HasPrefix(name, ".") || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}

		// Only process .go files (skip test files for now)
		if filepath.Ext(path) == ".go" && !strings.HasSuffix(path, "_test.go") {
			if err := ca.AnalyzeGoFile(ctx, path); err != nil {
				// Log error but continue
				return nil
			}
		}

		return nil
	})
}

// GetSymbolsByPackage returns all symbols for a package
func (ca *CodeAnalyzer) GetSymbolsByPackage(pkg string) []Symbol {
	return ca.symbols[pkg]
}

// GetSymbolsByType returns all symbols of a specific type
func (ca *CodeAnalyzer) GetSymbolsByType(symbolType string) []Symbol {
	result := make([]Symbol, 0)
	for _, symbols := range ca.symbols {
		for _, sym := range symbols {
			if sym.Type == symbolType {
				result = append(result, sym)
			}
		}
	}
	return result
}

// FindSymbol finds a symbol by name across all packages
func (ca *CodeAnalyzer) FindSymbol(name string) []Symbol {
	result := make([]Symbol, 0)
	for _, symbols := range ca.symbols {
		for _, sym := range symbols {
			if sym.Name == name {
				result = append(result, sym)
			}
		}
	}
	return result
}

// GetImports returns imports for a file
func (ca *CodeAnalyzer) GetImports(file string) []string {
	return ca.imports[file]
}

// GetDependencies returns all unique imported packages
func (ca *CodeAnalyzer) GetDependencies() []string {
	deps := make(map[string]bool)
	for _, imports := range ca.imports {
		for _, imp := range imports {
			deps[imp] = true
		}
	}
	
	result := make([]string, 0, len(deps))
	for dep := range deps {
		result = append(result, dep)
	}
	return result
}

// GetPackages returns all analyzed packages
func (ca *CodeAnalyzer) GetPackages() []string {
	pkgs := make([]string, 0, len(ca.symbols))
	for pkg := range ca.symbols {
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}

// GetSymbolCount returns total number of symbols
func (ca *CodeAnalyzer) GetSymbolCount() int {
	count := 0
	for _, symbols := range ca.symbols {
		count += len(symbols)
	}
	return count
}

// Helper function to convert AST expression to string
func exprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.StarExpr:
		return "*" + exprToString(e.X)
	case *ast.ArrayType:
		return "[]" + exprToString(e.Elt)
	case *ast.MapType:
		return "map[" + exprToString(e.Key) + "]" + exprToString(e.Value)
	case *ast.SelectorExpr:
		return exprToString(e.X) + "." + e.Sel.Name
	case *ast.InterfaceType:
		return "interface{}"
	default:
		return "unknown"
	}
}
