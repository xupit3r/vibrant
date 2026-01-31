package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/xupit3r/vibrant/internal/codeintel"
)

// CodeAnalysisTool analyzes code structure and symbols
type CodeAnalysisTool struct {
	analyzer *codeintel.CodeAnalyzer
}

// NewCodeAnalysisTool creates a new code analysis tool
func NewCodeAnalysisTool() *CodeAnalysisTool {
	return &CodeAnalysisTool{
		analyzer: codeintel.NewCodeAnalyzer(),
	}
}

func (c *CodeAnalysisTool) Name() string {
	return "analyze_code"
}

func (c *CodeAnalysisTool) Description() string {
	return "Analyze Go code structure, extract symbols, and track dependencies"
}

func (c *CodeAnalysisTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "File or directory path to analyze",
			Type:        "string",
			Required:    true,
		},
		"query": {
			Name:        "query",
			Description: "Query type: 'symbols', 'functions', 'types', 'imports', 'find:<name>'",
			Type:        "string",
			Required:    false,
		},
	}
}

func (c *CodeAnalysisTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, ok := params["path"].(string)
	if !ok || path == "" {
		return &Result{
			Success: false,
			Error:   "path parameter is required",
		}, nil
	}

	query, _ := params["query"].(string)
	if query == "" {
		query = "symbols"
	}

	// Analyze the path
	var err error
	if strings.HasSuffix(path, ".go") {
		err = c.analyzer.AnalyzeGoFile(ctx, path)
	} else {
		err = c.analyzer.AnalyzeGoDirectory(ctx, path)
	}

	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("analysis failed: %v", err),
		}, nil
	}

	// Process query
	output := ""
	switch {
	case query == "symbols":
		output = c.formatAllSymbols()
	case query == "functions":
		output = c.formatSymbolsByType("function")
	case query == "methods":
		output = c.formatSymbolsByType("method")
	case query == "types":
		output = c.formatTypes()
	case query == "imports":
		output = c.formatDependencies()
	case strings.HasPrefix(query, "find:"):
		name := strings.TrimPrefix(query, "find:")
		output = c.formatFindSymbol(strings.TrimSpace(name))
	default:
		output = c.formatSummary()
	}

	return &Result{
		Success: true,
		Output:  output,
	}, nil
}

func (c *CodeAnalysisTool) formatAllSymbols() string {
	var builder strings.Builder
	builder.WriteString("Code Symbols:\n\n")

	for _, pkg := range c.analyzer.GetPackages() {
		symbols := c.analyzer.GetSymbolsByPackage(pkg)
		if len(symbols) > 0 {
			builder.WriteString(fmt.Sprintf("Package: %s\n", pkg))
			for _, sym := range symbols {
				builder.WriteString(fmt.Sprintf("  %s %s (line %d)\n", sym.Type, sym.Name, sym.Line))
				if sym.Signature != "" {
					builder.WriteString(fmt.Sprintf("    %s\n", sym.Signature))
				}
			}
			builder.WriteString("\n")
		}
	}

	return builder.String()
}

func (c *CodeAnalysisTool) formatSymbolsByType(symbolType string) string {
	symbols := c.analyzer.GetSymbolsByType(symbolType)
	if len(symbols) == 0 {
		return fmt.Sprintf("No %ss found\n", symbolType)
	}

	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("%ss (%d):\n\n", strings.Title(symbolType), len(symbols)))

	for _, sym := range symbols {
		builder.WriteString(fmt.Sprintf("  %s (package %s, line %d)\n", sym.Name, sym.Package, sym.Line))
		if sym.Signature != "" {
			builder.WriteString(fmt.Sprintf("    %s\n", sym.Signature))
		}
		if sym.Receiver != "" {
			builder.WriteString(fmt.Sprintf("    Receiver: %s\n", sym.Receiver))
		}
	}

	return builder.String()
}

func (c *CodeAnalysisTool) formatTypes() string {
	var builder strings.Builder
	builder.WriteString("Types:\n\n")

	for _, typeName := range []string{"struct", "interface", "type"} {
		types := c.analyzer.GetSymbolsByType(typeName)
		if len(types) > 0 {
			builder.WriteString(fmt.Sprintf("%ss (%d):\n", strings.Title(typeName), len(types)))
			for _, t := range types {
				builder.WriteString(fmt.Sprintf("  %s (package %s, line %d)\n", t.Name, t.Package, t.Line))
			}
			builder.WriteString("\n")
		}
	}

	return builder.String()
}

func (c *CodeAnalysisTool) formatDependencies() string {
	deps := c.analyzer.GetDependencies()
	if len(deps) == 0 {
		return "No external dependencies found\n"
	}

	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("Dependencies (%d):\n\n", len(deps)))

	for _, dep := range deps {
		builder.WriteString(fmt.Sprintf("  - %s\n", dep))
	}

	return builder.String()
}

func (c *CodeAnalysisTool) formatFindSymbol(name string) string {
	symbols := c.analyzer.FindSymbol(name)
	if len(symbols) == 0 {
		return fmt.Sprintf("Symbol '%s' not found\n", name)
	}

	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("Found '%s' (%d matches):\n\n", name, len(symbols)))

	for _, sym := range symbols {
		builder.WriteString(fmt.Sprintf("  %s %s\n", sym.Type, sym.Name))
		builder.WriteString(fmt.Sprintf("    Package: %s\n", sym.Package))
		builder.WriteString(fmt.Sprintf("    File: %s (line %d)\n", sym.File, sym.Line))
		if sym.Signature != "" {
			builder.WriteString(fmt.Sprintf("    Signature: %s\n", sym.Signature))
		}
		builder.WriteString("\n")
	}

	return builder.String()
}

func (c *CodeAnalysisTool) formatSummary() string {
	var builder strings.Builder
	builder.WriteString("Code Analysis Summary:\n\n")

	builder.WriteString(fmt.Sprintf("Packages: %d\n", len(c.analyzer.GetPackages())))
	builder.WriteString(fmt.Sprintf("Total Symbols: %d\n", c.analyzer.GetSymbolCount()))
	builder.WriteString(fmt.Sprintf("Functions: %d\n", len(c.analyzer.GetSymbolsByType("function"))))
	builder.WriteString(fmt.Sprintf("Methods: %d\n", len(c.analyzer.GetSymbolsByType("method"))))
	builder.WriteString(fmt.Sprintf("Structs: %d\n", len(c.analyzer.GetSymbolsByType("struct"))))
	builder.WriteString(fmt.Sprintf("Interfaces: %d\n", len(c.analyzer.GetSymbolsByType("interface"))))
	builder.WriteString(fmt.Sprintf("Dependencies: %d\n", len(c.analyzer.GetDependencies())))

	return builder.String()
}
