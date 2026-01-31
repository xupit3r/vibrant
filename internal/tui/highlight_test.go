package tui

import (
	"strings"
	"testing"
)

func TestHighlightCode(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		contains []string // strings that should be in output
	}{
		{
			name: "Go code block",
			input: "```go\npackage main\n\nfunc main() {\n}\n```",
			contains: []string{"package", "main", "func"},
		},
		{
			name: "Python code block",
			input: "```python\ndef hello():\n    print('hello')\n```",
			contains: []string{"def", "hello"},
		},
		{
			name: "No language specified",
			input: "```\nsome code\n```",
			contains: []string{"some code"},
		},
		{
			name: "Text without code blocks",
			input: "This is plain text",
			contains: []string{"This is plain text"},
		},
		{
			name: "Multiple code blocks",
			input: "Text\n```go\nvar x int\n```\nMore text\n```python\ny = 1\n```",
			contains: []string{"var", "x", "y", "="},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HighlightCode(tt.input)
			for _, expected := range tt.contains {
				if !strings.Contains(result, expected) {
					t.Errorf("Expected output to contain %q", expected)
				}
			}
		})
	}
}

func TestStripANSI(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "ANSI color codes",
			input:    "\x1b[31mRed\x1b[0m text",
			expected: "Red text",
		},
		{
			name:     "No ANSI codes",
			input:    "Plain text",
			expected: "Plain text",
		},
		{
			name:     "Multiple ANSI codes",
			input:    "\x1b[1m\x1b[31mBold Red\x1b[0m\x1b[0m",
			expected: "Bold Red",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := StripANSI(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestFormatCodeBlock(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		language string
		contains []string
	}{
		{
			name:     "Simple code block",
			code:     "package main",
			language: "go",
			contains: []string{"┌─ go ─", "│ package main", "└─"},
		},
		{
			name:     "Multi-line code",
			code:     "line1\nline2\nline3",
			language: "text",
			contains: []string{"│ line1", "│ line2", "│ line3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatCodeBlock(tt.code, tt.language)
			for _, expected := range tt.contains {
				if !strings.Contains(result, expected) {
					t.Errorf("Expected output to contain %q, got:\n%s", expected, result)
				}
			}
		})
	}
}

func TestHighlightCodeBlock(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		language string
	}{
		{
			name:     "Go code",
			code:     "package main\nfunc main() {}",
			language: "go",
		},
		{
			name:     "Python code",
			code:     "def hello():\n    pass",
			language: "python",
		},
		{
			name:     "Unknown language",
			code:     "some text",
			language: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := highlightCodeBlock(tt.code, tt.language)
			// Should at least return something (even if unhighlighted)
			if result == "" {
				t.Error("Expected non-empty result")
			}
		})
	}
}
