package assistant

import (
	"strings"
	"testing"
)

func TestGetTemplate(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{"default", "default"},
		{"concise", "concise"},
		{"detailed", "detailed"},
		{"debug", "debug"},
		{"review", "review"},
		{"unknown", "default"}, // Should fallback to default
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			template := GetTemplate(tt.name)
			if template.Name != tt.expected {
				t.Errorf("GetTemplate(%s).Name = %s; want %s", 
					tt.name, template.Name, tt.expected)
			}
			
			if template.System == "" {
				t.Error("Template system prompt should not be empty")
			}
		})
	}
}

func TestListTemplates(t *testing.T) {
	templates := ListTemplates()
	
	if len(templates) == 0 {
		t.Error("Expected at least one template")
	}
	
	expectedTemplates := []string{"default", "concise", "detailed", "debug", "review"}
	if len(templates) != len(expectedTemplates) {
		t.Errorf("Expected %d templates, got %d", len(expectedTemplates), len(templates))
	}
	
	for _, expected := range expectedTemplates {
		found := false
		for _, tmpl := range templates {
			if tmpl == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Template %s not found in list", expected)
		}
	}
}

func TestNewPromptBuilder(t *testing.T) {
	pb := NewPromptBuilder("concise")
	
	if pb.template.Name != "concise" {
		t.Errorf("Expected template name 'concise', got '%s'", pb.template.Name)
	}
}

func TestBuildSimplePrompt(t *testing.T) {
	pb := NewPromptBuilder("default")
	
	prompt := pb.BuildSimplePrompt("What is Go?")
	
	if prompt == "" {
		t.Error("Prompt should not be empty")
	}
	
	if !strings.Contains(prompt, "What is Go?") {
		t.Error("Prompt should contain the user query")
	}
	
	if !strings.Contains(prompt, DefaultTemplate.System) {
		t.Error("Prompt should contain system message")
	}
}

func TestBuildWithContext(t *testing.T) {
	pb := NewPromptBuilder("default")
	
	context := "File: main.go\nContent: package main"
	prompt := pb.BuildWithContext("Explain this code", context)
	
	if !strings.Contains(prompt, "Explain this code") {
		t.Error("Prompt should contain the user query")
	}
	
	if !strings.Contains(prompt, context) {
		t.Error("Prompt should contain the context")
	}
	
	if !strings.Contains(prompt, "<context>") {
		t.Error("Prompt should contain context tags")
	}
}

func TestBuildWithHistory(t *testing.T) {
	pb := NewPromptBuilder("default")
	
	history := "[user]: Hello\n[assistant]: Hi there"
	prompt := pb.BuildWithHistory("Tell me more", history)
	
	if !strings.Contains(prompt, "Tell me more") {
		t.Error("Prompt should contain the user query")
	}
	
	if !strings.Contains(prompt, history) {
		t.Error("Prompt should contain conversation history")
	}
	
	if !strings.Contains(prompt, "Conversation History") {
		t.Error("Prompt should label the history section")
	}
}

func TestBuildPromptComplete(t *testing.T) {
	pb := NewPromptBuilder("detailed")
	
	query := "How do I use channels?"
	context := "File: main.go\nCode using goroutines"
	history := "[user]: What are goroutines?\n[assistant]: Concurrent functions"
	
	prompt := pb.BuildPrompt(query, context, history)
	
	// Should contain all components
	if !strings.Contains(prompt, query) {
		t.Error("Prompt missing query")
	}
	
	if !strings.Contains(prompt, context) {
		t.Error("Prompt missing context")
	}
	
	if !strings.Contains(prompt, history) {
		t.Error("Prompt missing history")
	}
	
	if !strings.Contains(prompt, DetailedTemplate.System) {
		t.Error("Prompt missing system message")
	}
}

func TestTemplateConsistency(t *testing.T) {
	templates := []string{"default", "concise", "detailed", "debug", "review"}
	
	for _, name := range templates {
		t.Run(name, func(t *testing.T) {
			template := GetTemplate(name)
			
			if template.Name != name {
				t.Errorf("Template name mismatch: expected %s, got %s", 
					name, template.Name)
			}
			
			if template.System == "" {
				t.Error("System prompt should not be empty")
			}
			
			if template.UserPrefix == "" {
				t.Error("User prefix should not be empty")
			}
			
			if template.AssistPrefix == "" {
				t.Error("Assistant prefix should not be empty")
			}
			
			if template.ContextFmt == "" {
				t.Error("Context format should not be empty")
			}
			
			// Context format should contain %s placeholder
			if !strings.Contains(template.ContextFmt, "%%s") && !strings.Contains(template.ContextFmt, "%s") {
				t.Error("Context format should contain percentage-s placeholder")
			}
		})
	}
}

func TestPromptBuilderDifferentTemplates(t *testing.T) {
	query := "Test question"
	
	templates := []string{"default", "concise", "detailed"}
	prompts := make(map[string]string)
	
	for _, tmpl := range templates {
		pb := NewPromptBuilder(tmpl)
		prompts[tmpl] = pb.BuildSimplePrompt(query)
	}
	
	// All prompts should contain the query
	for tmpl, prompt := range prompts {
		if !strings.Contains(prompt, query) {
			t.Errorf("Prompt for template %s missing query", tmpl)
		}
	}
	
	// Prompts should be different (different templates)
	if prompts["default"] == prompts["concise"] {
		t.Error("Default and concise prompts should be different")
	}
	
	if prompts["default"] == prompts["detailed"] {
		t.Error("Default and detailed prompts should be different")
	}
}
