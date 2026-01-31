package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/xupit3r/vibrant/internal/diff"
)

// ApplyDiffTool applies a diff/patch to a file
type ApplyDiffTool struct{}

func (a *ApplyDiffTool) Name() string {
	return "apply_diff"
}

func (a *ApplyDiffTool) Description() string {
	return "Apply a unified diff patch to a file"
}

func (a *ApplyDiffTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"file": {
			Name:        "file",
			Description: "Target file path",
			Type:        "string",
			Required:    true,
		},
		"diff": {
			Name:        "diff",
			Description: "Unified diff to apply",
			Type:        "string",
			Required:    true,
		},
	}
}

func (a *ApplyDiffTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	filePath, ok := params["file"].(string)
	if !ok || filePath == "" {
		return &Result{
			Success: false,
			Error:   "file parameter is required",
		}, nil
	}

	diffStr, ok := params["diff"].(string)
	if !ok || diffStr == "" {
		return &Result{
			Success: false,
			Error:   "diff parameter is required",
		}, nil
	}

	// Apply the diff using diff.Apply
	// Note: diff.Apply takes diffContent and targetFile path
	if err := diff.Apply(diffStr, filePath); err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to apply diff: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  fmt.Sprintf("Successfully applied diff to %s", filePath),
	}, nil
}

// GenerateDiffTool generates a diff between two versions of a file
type GenerateDiffTool struct{}

func (g *GenerateDiffTool) Name() string {
	return "generate_diff"
}

func (g *GenerateDiffTool) Description() string {
	return "Generate a unified diff between original and modified content"
}

func (g *GenerateDiffTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"original": {
			Name:        "original",
			Description: "Original content or file path",
			Type:        "string",
			Required:    true,
		},
		"modified": {
			Name:        "modified",
			Description: "Modified content or file path",
			Type:        "string",
			Required:    true,
		},
		"is_file": {
			Name:        "is_file",
			Description: "Whether inputs are file paths (default: false)",
			Type:        "bool",
			Required:    false,
		},
	}
}

func (g *GenerateDiffTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	original, ok := params["original"].(string)
	if !ok || original == "" {
		return &Result{
			Success: false,
			Error:   "original parameter is required",
		}, nil
	}

	modified, ok := params["modified"].(string)
	if !ok || modified == "" {
		return &Result{
			Success: false,
			Error:   "modified parameter is required",
		}, nil
	}

	isFile, _ := params["is_file"].(bool)

	// If is_file, read file contents
	if isFile {
		origContent, err := os.ReadFile(original)
		if err != nil {
			return &Result{
				Success: false,
				Error:   fmt.Sprintf("failed to read original file: %v", err),
			}, nil
		}
		original = string(origContent)

		modContent, err := os.ReadFile(modified)
		if err != nil {
			return &Result{
				Success: false,
				Error:   fmt.Sprintf("failed to read modified file: %v", err),
			}, nil
		}
		modified = string(modContent)
	}

	// Generate diff using diff.Generate
	fileDiff, err := diff.Generate(original, modified, "file")
	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to generate diff: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  fileDiff.FormatUnified(),
	}, nil
}

// BackupFileTool creates a backup of a file
type BackupFileTool struct{}

func (b *BackupFileTool) Name() string {
	return "backup_file"
}

func (b *BackupFileTool) Description() string {
	return "Create a backup copy of a file before modification"
}

func (b *BackupFileTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "File path to backup",
			Type:        "string",
			Required:    true,
		},
		"suffix": {
			Name:        "suffix",
			Description: "Backup suffix (default: .bak)",
			Type:        "string",
			Required:    false,
		},
	}
}

func (b *BackupFileTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, ok := params["path"].(string)
	if !ok || path == "" {
		return &Result{
			Success: false,
			Error:   "path parameter is required",
		}, nil
	}

	suffix, _ := params["suffix"].(string)
	if suffix == "" {
		suffix = ".bak"
	}

	// Read file
	content, err := os.ReadFile(path)
	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to read file: %v", err),
		}, nil
	}

	// Create backup
	backupPath := path + suffix
	if err := os.WriteFile(backupPath, content, 0644); err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to create backup: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  fmt.Sprintf("Backup created: %s", backupPath),
	}, nil
}

// ReplaceInFileTool performs find/replace operations in files
type ReplaceInFileTool struct{}

func (r *ReplaceInFileTool) Name() string {
	return "replace_in_file"
}

func (r *ReplaceInFileTool) Description() string {
	return "Find and replace text in a file"
}

func (r *ReplaceInFileTool) Parameters() map[string]Parameter {
	return map[string]Parameter{
		"path": {
			Name:        "path",
			Description: "File path",
			Type:        "string",
			Required:    true,
		},
		"find": {
			Name:        "find",
			Description: "Text to find",
			Type:        "string",
			Required:    true,
		},
		"replace": {
			Name:        "replace",
			Description: "Replacement text",
			Type:        "string",
			Required:    true,
		},
		"all": {
			Name:        "all",
			Description: "Replace all occurrences (default: true)",
			Type:        "bool",
			Required:    false,
		},
	}
}

func (r *ReplaceInFileTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
	path, ok := params["path"].(string)
	if !ok || path == "" {
		return &Result{
			Success: false,
			Error:   "path parameter is required",
		}, nil
	}

	find, ok := params["find"].(string)
	if !ok || find == "" {
		return &Result{
			Success: false,
			Error:   "find parameter is required",
		}, nil
	}

	replace, ok := params["replace"].(string)
	if !ok {
		replace = ""
	}

	replaceAll := true
	if all, ok := params["all"].(bool); ok {
		replaceAll = all
	}

	// Read file
	content, err := os.ReadFile(path)
	if err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to read file: %v", err),
		}, nil
	}

	// Perform replacement
	newContent := string(content)
	count := 0
	if replaceAll {
		count = strings.Count(newContent, find)
		newContent = strings.ReplaceAll(newContent, find, replace)
	} else {
		if strings.Contains(newContent, find) {
			newContent = strings.Replace(newContent, find, replace, 1)
			count = 1
		}
	}

	if count == 0 {
		return &Result{
			Success: true,
			Output:  "No matches found",
		}, nil
	}

	// Write file
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return &Result{
			Success: false,
			Error:   fmt.Sprintf("failed to write file: %v", err),
		}, nil
	}

	return &Result{
		Success: true,
		Output:  fmt.Sprintf("Replaced %d occurrence(s) in %s", count, filepath.Base(path)),
	}, nil
}
