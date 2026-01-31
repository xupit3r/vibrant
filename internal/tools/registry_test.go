package tools

import (
"context"
"testing"
)

// MockTool for testing
type MockTool struct {
name        string
description string
shouldFail  bool
}

func (m *MockTool) Name() string {
return m.name
}

func (m *MockTool) Description() string {
return m.description
}

func (m *MockTool) Parameters() map[string]Parameter {
return map[string]Parameter{
"input": {
Name:        "input",
Type:        "string",
Description: "Input parameter",
Required:    true,
},
}
}

func (m *MockTool) Execute(ctx context.Context, params map[string]interface{}) (*Result, error) {
if m.shouldFail {
return &Result{
Success: false,
Error:   "mock error",
}, nil
}

return &Result{
Success: true,
Output:  "mock output",
Data:    params,
}, nil
}

func TestNewRegistry(t *testing.T) {
r := NewRegistry()
if r == nil {
t.Fatal("NewRegistry returned nil")
}

if len(r.List()) != 0 {
t.Error("New registry should be empty")
}
}

func TestRegisterTool(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test tool"}

err := r.Register(tool)
if err != nil {
t.Fatalf("Failed to register tool: %v", err)
}

if len(r.List()) != 1 {
t.Errorf("Expected 1 tool, got %d", len(r.List()))
}
}

func TestRegisterDuplicateTool(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test tool"}

r.Register(tool)
err := r.Register(tool)

if err == nil {
t.Error("Expected error when registering duplicate tool")
}
}

func TestRegisterEmptyName(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "", description: "test"}

err := r.Register(tool)
if err == nil {
t.Error("Expected error when registering tool with empty name")
}
}

func TestGetTool(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test tool"}
r.Register(tool)

retrieved, err := r.Get("test")
if err != nil {
t.Fatalf("Failed to get tool: %v", err)
}

if retrieved.Name() != "test" {
t.Errorf("Expected tool name 'test', got '%s'", retrieved.Name())
}
}

func TestGetNonExistentTool(t *testing.T) {
r := NewRegistry()

_, err := r.Get("nonexistent")
if err == nil {
t.Error("Expected error when getting non-existent tool")
}
}

func TestListTools(t *testing.T) {
r := NewRegistry()

r.Register(&MockTool{name: "tool1", description: "Tool 1"})
r.Register(&MockTool{name: "tool2", description: "Tool 2"})
r.Register(&MockTool{name: "tool3", description: "Tool 3"})

tools := r.List()
if len(tools) != 3 {
t.Errorf("Expected 3 tools, got %d", len(tools))
}
}

func TestExecuteTool(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test tool"}
r.Register(tool)

ctx := context.Background()
params := map[string]interface{}{
"input": "test value",
}

result, err := r.Execute(ctx, "test", params)
if err != nil {
t.Fatalf("Failed to execute tool: %v", err)
}

if !result.Success {
t.Error("Expected successful execution")
}

if result.Output != "mock output" {
t.Errorf("Expected output 'mock output', got '%s'", result.Output)
}
}

func TestExecuteMissingRequiredParam(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test tool"}
r.Register(tool)

ctx := context.Background()
params := map[string]interface{}{} // Missing required "input" param

result, err := r.Execute(ctx, "test", params)
if err == nil {
t.Error("Expected error for missing required parameter")
}

if result == nil || result.Success {
t.Error("Result should indicate failure")
}
}

func TestGetToolsDescription(t *testing.T) {
r := NewRegistry()
r.Register(&MockTool{name: "tool1", description: "Tool 1"})
r.Register(&MockTool{name: "tool2", description: "Tool 2"})

desc := r.GetToolsDescription()

if desc == "" {
t.Error("Description should not be empty")
}

if len(desc) < 50 {
t.Error("Description seems too short")
}

// Should contain tool names
if !contains(desc, "tool1") || !contains(desc, "tool2") {
t.Error("Description should contain tool names")
}
}

func contains(s, substr string) bool {
return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && 
(s[:len(substr)] == substr || contains(s[1:], substr)))
}

func TestValidateParams(t *testing.T) {
r := NewRegistry()
tool := &MockTool{name: "test", description: "test"}

// Valid params
err := r.validateParams(tool, map[string]interface{}{
"input": "value",
})
if err != nil {
t.Errorf("Validation should pass for valid params: %v", err)
}

// Missing required param
err = r.validateParams(tool, map[string]interface{}{})
if err == nil {
t.Error("Validation should fail for missing required param")
}
}
