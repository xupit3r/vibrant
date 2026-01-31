package tools

import (
"context"
"fmt"
)

// Tool represents an action the agent can perform
type Tool interface {
// Name returns the tool name
Name() string

// Description returns what the tool does
Description() string

// Parameters returns the parameter schema
Parameters() map[string]Parameter

// Execute runs the tool with given parameters
Execute(ctx context.Context, params map[string]interface{}) (*Result, error)
}

// Parameter describes a tool parameter
type Parameter struct {
Name        string
Type        string // "string", "number", "boolean", "object", "array"
Description string
Required    bool
Default     interface{}
}

// Result represents the output of a tool execution
type Result struct {
Success bool
Output  string
Data    interface{}
Error   string
}

// Registry manages available tools
type Registry struct {
tools map[string]Tool
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
return &Registry{
tools: make(map[string]Tool),
}
}

// Register adds a tool to the registry
func (r *Registry) Register(tool Tool) error {
name := tool.Name()
if name == "" {
return fmt.Errorf("tool name cannot be empty")
}

if _, exists := r.tools[name]; exists {
return fmt.Errorf("tool %s already registered", name)
}

r.tools[name] = tool
return nil
}

// Get retrieves a tool by name
func (r *Registry) Get(name string) (Tool, error) {
tool, exists := r.tools[name]
if !exists {
return nil, fmt.Errorf("tool %s not found", name)
}
return tool, nil
}

// List returns all registered tools
func (r *Registry) List() []Tool {
tools := make([]Tool, 0, len(r.tools))
for _, tool := range r.tools {
tools = append(tools, tool)
}
return tools
}

// Execute runs a tool by name with parameters
func (r *Registry) Execute(ctx context.Context, name string, params map[string]interface{}) (*Result, error) {
tool, err := r.Get(name)
if err != nil {
return nil, err
}

// Validate parameters
if err := r.validateParams(tool, params); err != nil {
return &Result{
Success: false,
Error:   err.Error(),
}, err
}

return tool.Execute(ctx, params)
}

// validateParams validates parameters against tool schema
func (r *Registry) validateParams(tool Tool, params map[string]interface{}) error {
schema := tool.Parameters()

// Check required parameters
for name, param := range schema {
if param.Required {
if _, exists := params[name]; !exists {
return fmt.Errorf("required parameter %s missing", name)
}
}
}

// Type checking could be added here
return nil
}

// GetToolsDescription returns a description of all tools for LLM consumption
func (r *Registry) GetToolsDescription() string {
var desc string
desc += "Available Tools:\n\n"

for _, tool := range r.tools {
desc += fmt.Sprintf("## %s\n", tool.Name())
desc += fmt.Sprintf("%s\n\n", tool.Description())
desc += "Parameters:\n"
for _, param := range tool.Parameters() {
required := ""
if param.Required {
required = " (required)"
}
desc += fmt.Sprintf("  - %s (%s)%s: %s\n",
param.Name, param.Type, required, param.Description)
}
desc += "\n"
}

return desc
}
