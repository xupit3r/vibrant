package agent

import (
	"context"
	"fmt"
	"strings"
)

// Planner helps decompose goals into actionable steps
type Planner struct {
	agent *Agent
}

// NewPlanner creates a new task planner
func NewPlanner(agent *Agent) *Planner {
	return &Planner{
		agent: agent,
	}
}

// TaskDecomposition represents a broken-down task
type TaskDecomposition struct {
	Goal      string
	Subtasks  []Subtask
	Dependencies map[int][]int // subtask dependencies (index -> list of prerequisite indices)
}

// Subtask represents a single step in a larger task
type Subtask struct {
	ID          int
	Description string
	Actions     []Action
	Status      string // pending, in-progress, completed, failed
	Result      *ActionResult
}

// DecomposeTask breaks down a complex goal into subtasks
// This is a simple rule-based decomposition - in production this would use LLM
func (p *Planner) DecomposeTask(goal string) (*TaskDecomposition, error) {
	if goal == "" {
		return nil, fmt.Errorf("goal cannot be empty")
	}

	decomp := &TaskDecomposition{
		Goal:         goal,
		Subtasks:     []Subtask{},
		Dependencies: make(map[int][]int),
	}

	// Simple heuristic-based decomposition
	goalLower := strings.ToLower(goal)
	
	// Example: "read file X and search for pattern Y"
	if strings.Contains(goalLower, "read") && strings.Contains(goalLower, "search") {
		decomp.Subtasks = []Subtask{
			{
				ID:          0,
				Description: "Read the file",
				Status:      "pending",
			},
			{
				ID:          1,
				Description: "Search for pattern in content",
				Status:      "pending",
			},
		}
		decomp.Dependencies[1] = []int{0} // Task 1 depends on task 0
	}
	
	// Example: "create directory and write file"
	if strings.Contains(goalLower, "create") && strings.Contains(goalLower, "write") {
		decomp.Subtasks = []Subtask{
			{
				ID:          0,
				Description: "Create directory structure",
				Status:      "pending",
			},
			{
				ID:          1,
				Description: "Write file content",
				Status:      "pending",
			},
		}
		decomp.Dependencies[1] = []int{0}
	}

	if len(decomp.Subtasks) == 0 {
		// No decomposition pattern matched, create single task
		decomp.Subtasks = []Subtask{
			{
				ID:          0,
				Description: goal,
				Status:      "pending",
			},
		}
	}

	return decomp, nil
}

// ExecuteDecomposition executes a decomposed task respecting dependencies
func (p *Planner) ExecuteDecomposition(ctx context.Context, decomp *TaskDecomposition) error {
	if decomp == nil {
		return fmt.Errorf("decomposition is nil")
	}

	completed := make(map[int]bool)
	
	for len(completed) < len(decomp.Subtasks) {
		// Find next subtask that has all dependencies met
		nextTask := -1
		for i, subtask := range decomp.Subtasks {
			if completed[i] || subtask.Status == "completed" {
				continue
			}

			// Check if all dependencies are met
			deps := decomp.Dependencies[i]
			canExecute := true
			for _, depIdx := range deps {
				if !completed[depIdx] {
					canExecute = false
					break
				}
			}

			if canExecute {
				nextTask = i
				break
			}
		}

		if nextTask == -1 {
			// No more executable tasks - either all done or deadlock
			break
		}

		// Execute the subtask
		subtask := &decomp.Subtasks[nextTask]
		subtask.Status = "in-progress"

		if len(subtask.Actions) == 0 {
			// No actions defined, mark as completed
			subtask.Status = "completed"
			completed[nextTask] = true
			continue
		}

		// Create plan from actions
		plan := &Plan{
			Goal:    subtask.Description,
			Actions: subtask.Actions,
			Status:  "pending",
		}

		// Execute the plan
		_, err := p.agent.Execute(ctx, plan)
		if err != nil {
			subtask.Status = "failed"
			return fmt.Errorf("subtask %d failed: %v", nextTask, err)
		}

		subtask.Status = "completed"
		completed[nextTask] = true
	}

	// Check if all subtasks completed
	for i, subtask := range decomp.Subtasks {
		if !completed[i] && subtask.Status != "completed" {
			return fmt.Errorf("subtask %d not completed (possible dependency deadlock)", i)
		}
	}

	return nil
}

// SelfCorrector attempts to fix failed actions
type SelfCorrector struct {
	agent      *Agent
	maxRetries int
}

// NewSelfCorrector creates a new self-correction engine
func NewSelfCorrector(agent *Agent) *SelfCorrector {
	return &SelfCorrector{
		agent:      agent,
		maxRetries: 3,
	}
}

// SetMaxRetries sets the maximum number of retry attempts
func (s *SelfCorrector) SetMaxRetries(max int) {
	s.maxRetries = max
}

// CorrectAction attempts to fix a failed action
func (s *SelfCorrector) CorrectAction(ctx context.Context, action Action, failureReason string) (*Action, error) {
	// Simple correction strategies
	corrected := action

	reasonLower := strings.ToLower(failureReason)
	
	// Strategy 1: Path doesn't exist - try common alternatives
	if strings.Contains(reasonLower, "no such file") || strings.Contains(reasonLower, "not found") {
		if path, ok := action.Parameters["path"].(string); ok {
			// Try with ./ prefix
			if !strings.HasPrefix(path, "./") && !strings.HasPrefix(path, "/") {
				corrected.Parameters = make(map[string]interface{})
				for k, v := range action.Parameters {
					corrected.Parameters[k] = v
				}
				corrected.Parameters["path"] = "./" + path
				corrected.Reasoning = "Retrying with relative path prefix"
				return &corrected, nil
			}
		}
	}

	// Strategy 2: Permission denied - try with different permissions
	if strings.Contains(reasonLower, "permission denied") {
		corrected.Reasoning = "Permission issue - cannot auto-correct"
		return nil, fmt.Errorf("permission denied - manual intervention required")
	}

	// Strategy 3: Command not found - suggest alternatives
	if strings.Contains(reasonLower, "command not found") {
		return nil, fmt.Errorf("command not found - manual intervention required")
	}

	// No correction strategy found
	return nil, fmt.Errorf("no correction strategy available for: %s", failureReason)
}

// ExecuteWithCorrection executes an action with self-correction on failure
func (s *SelfCorrector) ExecuteWithCorrection(ctx context.Context, action Action) (*ActionResult, error) {
	var lastResult *ActionResult
	var lastErr error

	for attempt := 0; attempt <= s.maxRetries; attempt++ {
		// Execute the action
		result, err := s.agent.ExecuteSingle(ctx, action)
		lastResult = result
		lastErr = err

		// If successful, return
		if err == nil && result.Success {
			return result, nil
		}

		// If this was the last attempt, break
		if attempt == s.maxRetries {
			break
		}

		// Attempt correction
		failureReason := ""
		if err != nil {
			failureReason = err.Error()
		} else if result != nil && result.Result != nil {
			failureReason = result.Result.Error
		}

		corrected, corrErr := s.CorrectAction(ctx, action, failureReason)
		if corrErr != nil {
			// No correction possible
			return lastResult, fmt.Errorf("correction failed: %v", corrErr)
		}

		// Use corrected action for next attempt
		action = *corrected
	}

	return lastResult, fmt.Errorf("action failed after %d attempts: %v", s.maxRetries+1, lastErr)
}

// ExecutePlanWithCorrection executes a plan with self-correction
func (s *SelfCorrector) ExecutePlanWithCorrection(ctx context.Context, plan *Plan) ([]ActionResult, error) {
	if plan == nil || len(plan.Actions) == 0 {
		return nil, fmt.Errorf("plan is empty")
	}

	plan.Status = "executing"
	results := make([]ActionResult, 0, len(plan.Actions))

	for i, action := range plan.Actions {
		// Execute with correction
		result, err := s.ExecuteWithCorrection(ctx, action)
		
		if result != nil {
			results = append(results, *result)
		}

		if err != nil {
			plan.Status = "failed"
			return results, fmt.Errorf("action %d failed: %v", i+1, err)
		}
	}

	plan.Status = "completed"
	return results, nil
}
