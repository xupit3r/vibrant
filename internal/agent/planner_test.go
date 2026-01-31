package agent

import (
	"context"
	"testing"

	"github.com/xupit3r/vibrant/internal/tools"
)

func TestNewPlanner(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	planner := NewPlanner(agent)

	if planner == nil {
		t.Fatal("NewPlanner returned nil")
	}

	if planner.agent != agent {
		t.Error("Planner agent not set correctly")
	}
}

func TestDecomposeTask(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	planner := NewPlanner(agent)

	tests := []struct {
		name            string
		goal            string
		expectErr       bool
		expectedSubtasks int
		hasDependencies bool
	}{
		{
			name:            "read and search",
			goal:            "read file test.txt and search for pattern 'hello'",
			expectErr:       false,
			expectedSubtasks: 2,
			hasDependencies: true,
		},
		{
			name:            "create and write",
			goal:            "create directory test and write file config.json",
			expectErr:       false,
			expectedSubtasks: 2,
			hasDependencies: true,
		},
		{
			name:            "simple task",
			goal:            "list all files in current directory",
			expectErr:       false,
			expectedSubtasks: 1,
			hasDependencies: false,
		},
		{
			name:            "empty goal",
			goal:            "",
			expectErr:       true,
			expectedSubtasks: 0,
			hasDependencies: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decomp, err := planner.DecomposeTask(tt.goal)

			if tt.expectErr && err == nil {
				t.Error("Expected error but got none")
			}

			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !tt.expectErr {
				if len(decomp.Subtasks) != tt.expectedSubtasks {
					t.Errorf("Expected %d subtasks, got %d", tt.expectedSubtasks, len(decomp.Subtasks))
				}

				if tt.hasDependencies && len(decomp.Dependencies) == 0 {
					t.Error("Expected dependencies but got none")
				}

				// Check that all subtasks have valid IDs
				for i, subtask := range decomp.Subtasks {
					if subtask.ID != i {
						t.Errorf("Subtask %d has incorrect ID %d", i, subtask.ID)
					}
					if subtask.Status != "pending" {
						t.Errorf("Subtask %d has incorrect initial status: %s", i, subtask.Status)
					}
				}
			}
		})
	}
}

func TestExecuteDecomposition(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	planner := NewPlanner(agent)

	// Create a simple decomposition with actions
	decomp := &TaskDecomposition{
		Goal: "Test decomposition",
		Subtasks: []Subtask{
			{
				ID:          0,
				Description: "First task",
				Status:      "pending",
				Actions: []Action{
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test1"}},
				},
			},
			{
				ID:          1,
				Description: "Second task",
				Status:      "pending",
				Actions: []Action{
					{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test2"}},
				},
			},
		},
		Dependencies: map[int][]int{
			1: {0}, // Task 1 depends on task 0
		},
	}

	err := planner.ExecuteDecomposition(context.Background(), decomp)
	if err != nil {
		t.Fatalf("ExecuteDecomposition failed: %v", err)
	}

	// Check that all subtasks completed
	for i, subtask := range decomp.Subtasks {
		if subtask.Status != "completed" {
			t.Errorf("Subtask %d not completed: %s", i, subtask.Status)
		}
	}
}

func TestExecuteDecompositionWithFailure(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "failing_tool", description: "Failing tool", shouldFail: true}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	planner := NewPlanner(agent)

	decomp := &TaskDecomposition{
		Goal: "Test failure",
		Subtasks: []Subtask{
			{
				ID:          0,
				Description: "Failing task",
				Status:      "pending",
				Actions: []Action{
					{Tool: "failing_tool", Parameters: map[string]interface{}{"input": "test"}},
				},
			},
		},
		Dependencies: make(map[int][]int),
	}

	err := planner.ExecuteDecomposition(context.Background(), decomp)
	if err == nil {
		t.Error("Expected ExecuteDecomposition to fail")
	}

	if decomp.Subtasks[0].Status != "failed" {
		t.Errorf("Expected subtask status 'failed', got '%s'", decomp.Subtasks[0].Status)
	}
}

func TestExecuteDecompositionNil(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	planner := NewPlanner(agent)

	err := planner.ExecuteDecomposition(context.Background(), nil)
	if err == nil {
		t.Error("Expected error for nil decomposition")
	}
}

func TestNewSelfCorrector(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	if corrector == nil {
		t.Fatal("NewSelfCorrector returned nil")
	}

	if corrector.agent != agent {
		t.Error("SelfCorrector agent not set correctly")
	}

	if corrector.maxRetries != 3 {
		t.Errorf("Expected default maxRetries=3, got %d", corrector.maxRetries)
	}
}

func TestSetMaxRetries(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	corrector.SetMaxRetries(5)
	if corrector.maxRetries != 5 {
		t.Errorf("Expected maxRetries=5, got %d", corrector.maxRetries)
	}
}

func TestCorrectActionFileNotFound(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	action := Action{
		Tool: "read_file",
		Parameters: map[string]interface{}{
			"path": "test.txt",
		},
	}

	corrected, err := corrector.CorrectAction(context.Background(), action, "no such file or directory")
	if err != nil {
		t.Fatalf("CorrectAction failed: %v", err)
	}

	// Should add ./ prefix
	if corrected.Parameters["path"] != "./test.txt" {
		t.Errorf("Expected path './test.txt', got '%s'", corrected.Parameters["path"])
	}
}

func TestCorrectActionPermissionDenied(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	action := Action{
		Tool: "write_file",
		Parameters: map[string]interface{}{
			"path": "/root/test.txt",
		},
	}

	_, err := corrector.CorrectAction(context.Background(), action, "permission denied")
	if err == nil {
		t.Error("Expected error for permission denied")
	}
}

func TestCorrectActionNoStrategy(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	action := Action{
		Tool:       "test_tool",
		Parameters: map[string]interface{}{},
	}

	_, err := corrector.CorrectAction(context.Background(), action, "unknown error")
	if err == nil {
		t.Error("Expected error when no correction strategy available")
	}
}

func TestExecuteWithCorrection(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	corrector := NewSelfCorrector(agent)

	action := Action{
		Tool:       "test_tool",
		Parameters: map[string]interface{}{"input": "test"},
	}

	result, err := corrector.ExecuteWithCorrection(context.Background(), action)
	if err != nil {
		t.Fatalf("ExecuteWithCorrection failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected action to succeed")
	}
}

func TestExecuteWithCorrectionMaxRetries(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "failing_tool", description: "Failing tool", shouldFail: true}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	corrector := NewSelfCorrector(agent)
	corrector.SetMaxRetries(2)

	action := Action{
		Tool:       "failing_tool",
		Parameters: map[string]interface{}{"input": "test"},
	}

	result, err := corrector.ExecuteWithCorrection(context.Background(), action)
	if err == nil {
		t.Error("Expected error after max retries")
	}

	if result.Success {
		t.Error("Expected action to fail")
	}
}

func TestExecutePlanWithCorrection(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "test_tool", description: "Test tool"}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	corrector := NewSelfCorrector(agent)

	plan := &Plan{
		Goal: "Test with correction",
		Actions: []Action{
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test1"}},
			{Tool: "test_tool", Parameters: map[string]interface{}{"input": "test2"}},
		},
		Status: "pending",
	}

	results, err := corrector.ExecutePlanWithCorrection(context.Background(), plan)
	if err != nil {
		t.Fatalf("ExecutePlanWithCorrection failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	if plan.Status != "completed" {
		t.Errorf("Expected status 'completed', got '%s'", plan.Status)
	}
}

func TestExecutePlanWithCorrectionFailure(t *testing.T) {
	registry := tools.NewRegistry()
	mockTool := &mockTool{name: "failing_tool", description: "Failing tool", shouldFail: true}
	registry.Register(mockTool)

	agent := NewAgent(registry)
	corrector := NewSelfCorrector(agent)
	corrector.SetMaxRetries(1)

	plan := &Plan{
		Goal: "Test failure with correction",
		Actions: []Action{
			{Tool: "failing_tool", Parameters: map[string]interface{}{"input": "test"}},
		},
		Status: "pending",
	}

	_, err := corrector.ExecutePlanWithCorrection(context.Background(), plan)
	if err == nil {
		t.Error("Expected error from failing plan")
	}

	if plan.Status != "failed" {
		t.Errorf("Expected status 'failed', got '%s'", plan.Status)
	}
}

func TestExecutePlanWithCorrectionEmptyPlan(t *testing.T) {
	agent := NewAgent(tools.NewRegistry())
	corrector := NewSelfCorrector(agent)

	plan := &Plan{
		Goal:    "Empty plan",
		Actions: []Action{},
	}

	_, err := corrector.ExecutePlanWithCorrection(context.Background(), plan)
	if err == nil {
		t.Error("Expected error for empty plan")
	}
}
