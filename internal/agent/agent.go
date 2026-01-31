package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/xupit3r/vibrant/internal/tools"
)

// Action represents a single action the agent can take
type Action struct {
	Tool       string                 `json:"tool"`
	Parameters map[string]interface{} `json:"parameters"`
	Reasoning  string                 `json:"reasoning"`
}

// ActionResult contains the result of an action execution
type ActionResult struct {
	Action  Action
	Result  *tools.Result
	Success bool
}

// Plan represents a sequence of actions to accomplish a goal
type Plan struct {
	Goal    string   `json:"goal"`
	Actions []Action `json:"actions"`
	Status  string   `json:"status"` // pending, executing, completed, failed
}

// Agent orchestrates multi-step actions using tools
type Agent struct {
	registry *tools.Registry
	maxSteps int
	verbose  bool
}

// NewAgent creates a new agent with the given tool registry
func NewAgent(registry *tools.Registry) *Agent {
	return &Agent{
		registry: registry,
		maxSteps: 10, // Default max steps to prevent infinite loops
		verbose:  false,
	}
}

// SetMaxSteps sets the maximum number of steps the agent can take
func (a *Agent) SetMaxSteps(max int) {
	a.maxSteps = max
}

// SetVerbose enables/disables verbose logging
func (a *Agent) SetVerbose(verbose bool) {
	a.verbose = verbose
}

// Execute executes a plan step by step
func (a *Agent) Execute(ctx context.Context, plan *Plan) ([]ActionResult, error) {
	if plan == nil || len(plan.Actions) == 0 {
		return nil, fmt.Errorf("plan is empty")
	}

	if len(plan.Actions) > a.maxSteps {
		return nil, fmt.Errorf("plan exceeds maximum steps (%d > %d)", len(plan.Actions), a.maxSteps)
	}

	plan.Status = "executing"
	results := make([]ActionResult, 0, len(plan.Actions))

	for i, action := range plan.Actions {
		if a.verbose {
			fmt.Printf("Step %d/%d: %s (%s)\n", i+1, len(plan.Actions), action.Tool, action.Reasoning)
		}

		// Check context cancellation
		select {
		case <-ctx.Done():
			plan.Status = "cancelled"
			return results, ctx.Err()
		default:
		}

		// Execute the action
		result, err := a.registry.Execute(ctx, action.Tool, action.Parameters)
		actionResult := ActionResult{
			Action:  action,
			Result:  result,
			Success: err == nil && result.Success,
		}
		results = append(results, actionResult)

		if !actionResult.Success {
			plan.Status = "failed"
			if a.verbose {
				fmt.Printf("  ❌ Failed: %v\n", err)
			}
			return results, fmt.Errorf("action failed at step %d: %v", i+1, err)
		}

		if a.verbose {
			fmt.Printf("  ✅ Success: %s\n", result.Output)
		}
	}

	plan.Status = "completed"
	return results, nil
}

// ExecuteSingle executes a single action
func (a *Agent) ExecuteSingle(ctx context.Context, action Action) (*ActionResult, error) {
	result, err := a.registry.Execute(ctx, action.Tool, action.Parameters)
	actionResult := &ActionResult{
		Action:  action,
		Result:  result,
		Success: err == nil && result.Success,
	}

	if !actionResult.Success {
		return actionResult, fmt.Errorf("action failed: %v", err)
	}

	return actionResult, nil
}

// GetAvailableTools returns descriptions of all available tools
func (a *Agent) GetAvailableTools() string {
	return a.registry.GetToolsDescription()
}

// ValidateAction validates that an action can be executed
func (a *Agent) ValidateAction(action Action) error {
	tool, err := a.registry.Get(action.Tool)
	if err != nil {
		return fmt.Errorf("invalid tool: %v", err)
	}

	// Validate parameters
	params := tool.Parameters()
	for _, param := range params {
		if param.Required {
			if _, ok := action.Parameters[param.Name]; !ok {
				return fmt.Errorf("missing required parameter: %s", param.Name)
			}
		}
	}

	return nil
}

// ValidatePlan validates that a plan is executable
func (a *Agent) ValidatePlan(plan *Plan) error {
	if plan == nil {
		return fmt.Errorf("plan is nil")
	}

	if plan.Goal == "" {
		return fmt.Errorf("plan has no goal")
	}

	if len(plan.Actions) == 0 {
		return fmt.Errorf("plan has no actions")
	}

	if len(plan.Actions) > a.maxSteps {
		return fmt.Errorf("plan exceeds maximum steps (%d > %d)", len(plan.Actions), a.maxSteps)
	}

	// Validate each action
	for i, action := range plan.Actions {
		if err := a.ValidateAction(action); err != nil {
			return fmt.Errorf("invalid action at step %d: %v", i+1, err)
		}
	}

	return nil
}

// SummarizeResults creates a summary of action results
func (a *Agent) SummarizeResults(results []ActionResult) string {
	if len(results) == 0 {
		return "No actions executed"
	}

	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("Executed %d actions:\n", len(results)))

	successCount := 0
	for i, result := range results {
		status := "✅"
		if !result.Success {
			status = "❌"
		} else {
			successCount++
		}

		builder.WriteString(fmt.Sprintf("%d. %s %s: %s\n",
			i+1, status, result.Action.Tool, result.Action.Reasoning))

		if result.Result != nil && result.Result.Output != "" {
			// Truncate long outputs
			output := result.Result.Output
			if len(output) > 100 {
				output = output[:97] + "..."
			}
			builder.WriteString(fmt.Sprintf("   Output: %s\n", output))
		}
	}

	builder.WriteString(fmt.Sprintf("\nSuccess rate: %d/%d", successCount, len(results)))
	return builder.String()
}
