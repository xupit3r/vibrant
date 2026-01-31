package model

import (
	"testing"
)

func TestGetModelByID(t *testing.T) {
	tests := []struct {
		id        string
		shouldErr bool
	}{
		{"qwen2.5-coder-3b-q4", false},
		{"qwen2.5-coder-7b-q5", false},
		{"qwen2.5-coder-14b-q5", false},
		{"nonexistent-model", true},
		{"", true},
	}
	
	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			model, err := GetModelByID(tt.id)
			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error for ID %s, got nil", tt.id)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for ID %s: %v", tt.id, err)
				}
				if model == nil {
					t.Errorf("Expected model for ID %s, got nil", tt.id)
				}
				if model.ID != tt.id {
					t.Errorf("Expected ID %s, got %s", tt.id, model.ID)
				}
			}
		})
	}
}

func TestFilterByRAM(t *testing.T) {
	tests := []struct {
		availableBytes int64
		expectedCount  int
		description    string
	}{
		{2 * 1024 * 1024 * 1024, 0, "2GB - too small"},
		{6 * 1024 * 1024 * 1024, 1, "6GB - should get 3B model"},
		{12 * 1024 * 1024 * 1024, 3, "12GB - should get 3B and 7B models"},
		{20 * 1024 * 1024 * 1024, 4, "20GB - should get all models"},
	}
	
	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			filtered := FilterByRAM(tt.availableBytes)
			if len(filtered) != tt.expectedCount {
				t.Errorf("Expected %d models, got %d for %s", 
					tt.expectedCount, len(filtered), tt.description)
			}
		})
	}
}

func TestFilterByTag(t *testing.T) {
	models := ListAll()
	
	recommended := FilterByTag(models, "recommended")
	if len(recommended) == 0 {
		t.Error("Expected at least one recommended model")
	}
	
	for _, m := range recommended {
		if !m.Recommended {
			t.Errorf("Model %s should be recommended", m.ID)
		}
	}
	
	coding := FilterByTag(models, "coding")
	if len(coding) == 0 {
		t.Error("Expected at least one coding model")
	}
}

func TestFilterRecommended(t *testing.T) {
	models := ListAll()
	recommended := FilterRecommended(models)
	
	if len(recommended) == 0 {
		t.Error("Expected at least one recommended model")
	}
	
	for _, m := range recommended {
		if !m.Recommended {
			t.Errorf("Model %s in recommended list but not marked as recommended", m.ID)
		}
	}
}

func TestSortByRAMRequired(t *testing.T) {
	models := ListAll()
	sorted := SortByRAMRequired(models)
	
	if len(sorted) != len(models) {
		t.Errorf("Expected %d models, got %d after sorting", len(models), len(sorted))
	}
	
	// Check descending order
	for i := 1; i < len(sorted); i++ {
		if sorted[i].RAMRequiredMB > sorted[i-1].RAMRequiredMB {
			t.Errorf("Models not sorted correctly: %s (%d MB) before %s (%d MB)",
				sorted[i-1].ID, sorted[i-1].RAMRequiredMB,
				sorted[i].ID, sorted[i].RAMRequiredMB)
		}
	}
}

func TestListAll(t *testing.T) {
	models := ListAll()
	
	if len(models) == 0 {
		t.Error("Expected at least one model in registry")
	}
	
	// Verify all models have required fields
	for _, m := range models {
		if m.ID == "" {
			t.Error("Model missing ID")
		}
		if m.Name == "" {
			t.Errorf("Model %s missing name", m.ID)
		}
		if m.RAMRequiredMB <= 0 {
			t.Errorf("Model %s has invalid RAM requirement: %d", m.ID, m.RAMRequiredMB)
		}
		if m.ContextWindow <= 0 {
			t.Errorf("Model %s has invalid context window: %d", m.ID, m.ContextWindow)
		}
	}
}
