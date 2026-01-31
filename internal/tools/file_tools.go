package tools

import (
"context"
"fmt"
"os"
"path/filepath"
)

// ReadFileTool reads file contents
type ReadFileTool struct{}

func (t *ReadFileTool) Name() string {
return "read_file"
}

func (t *ReadFileTool) Description() string {
return "Reads the contents of a file"
}

func (t *ReadFileTool) Parameters() map[string]Parameter {
return map[string]Parameter{
"path": {
Name:        "path",
Type:        "string",
Description: "Path to the file to read",
Required:    true,
},
}
}

func (t *ReadFileTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
path, ok := params["path"].(string)
if !ok {
return &Result{
Success: false,
Error:   "path must be a string",
}, fmt.Errorf("invalid path parameter")
}

content, err := os.ReadFile(path)
if err != nil {
return &Result{
Success: false,
Error:   err.Error(),
}, err
}

return &Result{
Success: true,
Output:  string(content),
Data:    map[string]interface{}{"size": len(content)},
}, nil
}

// WriteFileTool writes content to a file
type WriteFileTool struct{}

func (t *WriteFileTool) Name() string {
return "write_file"
}

func (t *WriteFileTool) Description() string {
return "Writes content to a file"
}

func (t *WriteFileTool) Parameters() map[string]Parameter {
return map[string]Parameter{
"path": {
Name:        "path",
Type:        "string",
Description: "Path to the file to write",
Required:    true,
},
"content": {
Name:        "content",
Type:        "string",
Description: "Content to write to the file",
Required:    true,
},
}
}

func (t *WriteFileTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
path, ok := params["path"].(string)
if !ok {
return &Result{
Success: false,
Error:   "path must be a string",
}, fmt.Errorf("invalid path parameter")
}

content, ok := params["content"].(string)
if !ok {
return &Result{
Success: false,
Error:   "content must be a string",
}, fmt.Errorf("invalid content parameter")
}

// Ensure directory exists
dir := filepath.Dir(path)
if err := os.MkdirAll(dir, 0755); err != nil {
return &Result{
Success: false,
Error:   err.Error(),
}, err
}

err := os.WriteFile(path, []byte(content), 0644)
if err != nil {
return &Result{
Success: false,
Error:   err.Error(),
}, err
}

return &Result{
Success: true,
Output:  fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), path),
Data:    map[string]interface{}{"bytes_written": len(content)},
}, nil
}

// ListDirectoryTool lists directory contents
type ListDirectoryTool struct{}

func (t *ListDirectoryTool) Name() string {
return "list_directory"
}

func (t *ListDirectoryTool) Description() string {
return "Lists the contents of a directory"
}

func (t *ListDirectoryTool) Parameters() map[string]Parameter {
return map[string]Parameter{
"path": {
Name:        "path",
Type:        "string",
Description: "Path to the directory to list",
Required:    true,
},
}
}

func (t *ListDirectoryTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
path, ok := params["path"].(string)
if !ok {
return &Result{
Success: false,
Error:   "path must be a string",
}, fmt.Errorf("invalid path parameter")
}

entries, err := os.ReadDir(path)
if err != nil {
return &Result{
Success: false,
Error:   err.Error(),
}, err
}

var files []string
for _, entry := range entries {
name := entry.Name()
if entry.IsDir() {
name += "/"
}
files = append(files, name)
}

output := fmt.Sprintf("Directory %s contains %d entries:\n", path, len(files))
for _, f := range files {
output += fmt.Sprintf("  %s\n", f)
}

return &Result{
Success: true,
Output:  output,
Data:    map[string]interface{}{"entries": files, "count": len(files)},
}, nil
}
