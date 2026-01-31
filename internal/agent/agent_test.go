package agent

import (
	"context"
	"testing"

	"github.com/xupit3r/vibrant/internal/tools"
)

// Mock tool for testing
type mockTool struct {
	name        string
	description string
	shouldFail  bool
	output      string
}

func (m *mockTool) Name() string {
	return m.name
}

func (m *mockTool) Description() string {
	return m.description
}

func (m *mockTool) Parameters() map[string]tools.Parameter {
	return map[string]tools.Parameter{
		"input": {
			Name:        "input",
			Description: "Test input",
			Type:        "string",
			Required:    true,
		},
	}
}

func (m *mockTool) Execute(ctx context.Context, params map[string]interface{}) (*tools.Result, error) {
	if m.shouldFail {
		return &tools.Result{
			Success: false,
			Error:   "mock error",
		}, nil
	}

	output := m.output
	if output == "" {
		output = "mock output"
	}

	return &tools.Result{
		Success: true,
		Output:  output,
	}, nil
}

func TestNewAgent(t *testing.T) {
	registry := tools.NewRegistry()
	agent := NewAgent(registry)

	if agent == nil {
		t.Fatal("NewAgent returned nil")
	}

	if agent.registry != registry {
		t.Error("Agent registry not set correctly")
	}

	if agent.maxSteps != 10 {
		t.Errorf("Expected default maxSteps=10, got %d", agent.maxSteps)
	}
}

func TestSetMaxSteps(t *testing.T) {
	registry := tools.NewRegistry()
	agent := NewAgent(registry)

	agent.SetMaxSteps(20)
	if agent.maxSteps != 20 {
		t.Errorf("Expected maxSteps=20, got %d", agent.maxSteps)
	}
}

func TestExecuteSingleAction(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	action := Action{
		Tool:       "test_tool",
		Parameters: map[string]interface{}{"input": "test"},
		Reasoning:  "Testing single action",
	}

	result, err := agent.ExecuteSingle(context.Background(), action)
	if err != nil {
		t.Fatalf("ExecuteSingle failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected action to succeed")
	}

	if result.Result.Output != "mock output" {
		t.Errorf("Expected output 'mock output', got '%s'", result.Result.Output)
	}
}

func TestExecuteSingleActionFailure(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "failing_tool", description: "Failing tool", shouldFail: true}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	action := Action{
		Tool:       "failing_tool",
		Parameters: map[string]interface{}{"input": "test"},
		Reasoning:  "Testing failure",
	}

	result, err := agent.ExecuteSingle(context.Background(), action)
	if err == nil {
		t.Error("Expected ExecuteSingle to return error")
	}

	if result.Success {
		t.Error("Expected action to fail")
	}
}

func TestExecutePlan(t *testing.T) {
	registry := tools.NewRegistry()
	tool1 := &mockTool{name: "tool1", description: "Tool 1", output: "output1"}
	tool2 := &mockTool{name: "tool2", description: "Tool 2", output: "output2"}
	registry.Register(tool1)
	registry.Register(tool2)

	agent := NewAgent(registry)
	plan := &Plan{
		Goal: "Test multi-step execution",
		Actions: []Action{
			{Tool: "tool1", Parameters: map[string]interface{}{"input": "test1"}, Reasoning: "Step 1"},
			{Tool: "tool2", Parameters: map[string]interface{}{"input": "test2"}, Reasoning: "Step 2"},
		},
		Status: "pending",
	}

	results, err := agent.Execute(context.Background(), plan)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	if plan.Status != "completed" {
		t.Errorf("Expected status 'completed', got '%s'", plan.Status)
	}

	for i, result := range results {
		if !result.Success {
			t.Errorf("Action %d failed", i+1)
		}
	}
}

func TestExecutePlanWithFailure(t *testing.T) {
	registry := tools.NewRegistry()
	tool1 := &mockTool{name: "tool1", description: "Tool 1"}
	tool2 := &mockTool{name: "tool2", description: "Tool 2", shouldFail: true}
	registry.Register(tool1)
	registry.Register(tool2)

	agent := NewAgent(registry)
	plan := &Plan{
		Goal: "Test failure handling",
		Actions: []Action{
			{Tool: "tool1", Parameters: map[string]interface{}{"input": "test1"}, Reasoning: "Step 1"},
			{Tool: "tool2", Parameters: map[string]interface{}{"input": "test2"}, Reasoning: "Step 2"},
		},
		Status: "pending",
	}

	results, err := agent.Execute(context.Background(), plan)
	if err == nil {
		t.Error("Expected Execute to return error")
	}

	if plan.Status != "failed" {
		t.Errorf("Expected status 'failed', got '%s'", plan.Status)
	}

	// Should only have results up to the failure
	if len(results) != 2 {
		t.Errorf("Expected 2 results (including failed one), got %d", len(results))
	}

	// First action should succeed
	if !results[0].Success {
		t.Error("First action should have succeeded")
	}

	// Second action should fail
	if results[1].Success {
		t.Error("Second action should have failed")
	}
}

func TestExecuteEmptyPlan(t *testing.T) {
	registry := tools.NewRegistry()
	agent := NewAgent(registry)

	plan := &Plan{
		Goal:    "Empty plan",
		Actions: []Action{},
	}

	_, err := agent.Execute(context.Background(), plan)
	if err == nil {
		t.Error("Expected error for empty plan")
	}
}

func TestExecuteExceedsMaxSteps(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	agent.SetMaxSteps(2)

	plan := &Plan{
		Goal: "Too many steps",
		Actions: []Action{
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "1"}, Reasoning: "Step 1"},
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "2"}, Reasoning: "Step 2"},
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "3"}, Reasoning: "Step 3"},
		},
	}

	_, err := agent.Execute(context.Background(), plan)
	if err == nil {
		t.Error("Expected error for exceeding max steps")
	}
}

func TestValidateAction(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)

	tests := []struct {
		name      string
		action    Action
		expectErr bool
	}{
		{
			name: "valid action",
			action: Action{
				Tool:       "test_tool",
				Parameters: map[string]interface{}{"input": "test"},
			},
			expectErr: false,
		},
		{
			name: "missing required parameter",
			action: Action{
				Tool:       "test_tool",
				Parameters: map[string]interface{}{},
			},
			expectErr: true,
		},
		{
			name: "non-existent tool",
			action: Action{
				Tool:       "nonexistent",
				Parameters: map[string]interface{}{"input": "test"},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := agent.ValidateAction(tt.action)
			if (err != nil) != tt.expectErr {
				t.Errorf("ValidateAction() error = %v, expectErr %v", err, tt.expectErr)
			}
		})
	}
}

func TestValidatePlan(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	agent.SetMaxSteps(2)

	tests := []struct {
		name      string
		plan      *Plan
		expectErr bool
	}{
		{
			name: "valid plan",
			plan: &Plan{
				Goal: "Test goal",
				Actions: []Action{
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test"}},
				},
			},
			expectErr: false,
		},
		{
			name:      "nil plan",
			plan:      nil,
			expectErr: true,
		},
		{
			name: "no goal",
			plan: &Plan{
				Goal: "",
				Actions: []Action{
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test"}},
				},
			},
			expectErr: true,
		},
		{
			name: "no actions",
			plan: &Plan{
				Goal:    "Test goal",
				Actions: []Action{},
			},
			expectErr: true,
		},
		{
			name: "exceeds max steps",
			plan: &Plan{
				Goal: "Test goal",
				Actions: []Action{
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "1"}},
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "2"}},
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "3"}},
				},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := agent.ValidatePlan(tt.plan)
			if (err != nil) != tt.expectErr {
				t.Errorf("ValidatePlan() error = %v, expectErr %v", err, tt.expectErr)
			}
		})
	}
}

func TestSummarizeResults(t *testing.T) {
	registry := tools.NewRegistry()
	agent := NewAgent(registry)

	results := []ActionResult{
		{
			Action:  Action{Tool: "tool1", Reasoning: "Test 1"},
			Result:  &tools.Result{Success: true, Output: "output1"},
			Success: true,
		},
		{
			Action:  Action{Tool: "tool2", Reasoning: "Test 2"},
			Result:  &tools.Result{Success: false, Error: "error"},
			Success: false,
		},
	}

	summary := agent.SummarizeResults(results)
	if summary == "" {
		t.Error("Summary should not be empty")
	}

	// Check that summary contains key information
	if !contains(summary, "tool1") || !contains(summary, "tool2") {
		t.Error("Summary should contain tool names")
	}

	if !contains(summary, "Success rate: 1/2") {
		t.Error("Summary should contain success rate")
	}
}

func TestSummarizeEmptyResults(t *testing.T) {
	registry := tools.NewRegistry()
	agent := NewAgent(registry)

	summary := agent.SummarizeResults([]ActionResult{})
	if summary != "No actions executed" {
		t.Errorf("Expected 'No actions executed', got '%s'", summary)
	}
}

func TestGetAvailableTools(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	description := agent.GetAvailableTools()

	if description == "" {
		t.Error("Available tools description should not be empty")
	}

	if !contains(description, "test_tool") {
		t.Error("Description should contain tool name")
	}
}

func TestContextCancellation(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	plan := &Plan{
		Goal: "Test cancellation",
		Actions: []Action{
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test"}},
		},
	}

	_, err := agent.Execute(ctx, plan)
	if err == nil {
		t.Error("Expected error from cancelled context")
	}

	if plan.Status != "cancelled" {
		t.Errorf("Expected status 'cancelled', got '%s'", plan.Status)
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && 
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		containsMiddle(s, substr)))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
