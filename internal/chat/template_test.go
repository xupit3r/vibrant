package chat

import (
	"strings"
	"testing"
)

func TestAutoDetectionChatML(t *testing.T) {
	raw := `{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}`
	ct := NewChatTemplate(raw)
	if ct.Type != ChatTemplateChatML {
		t.Errorf("expected ChatTemplateChatML, got %v", ct.Type)
	}
	if ct.StopToken != "<|im_end|>" {
		t.Errorf("expected stop token <|im_end|>, got %q", ct.StopToken)
	}
}

func TestAutoDetectionLlama3(t *testing.T) {
	raw := `{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}`
	ct := NewChatTemplate(raw)
	if ct.Type != ChatTemplateLlama3 {
		t.Errorf("expected ChatTemplateLlama3, got %v", ct.Type)
	}
	if ct.StopToken != "<|eot_id|>" {
		t.Errorf("expected stop token <|eot_id|>, got %q", ct.StopToken)
	}
}

func TestAutoDetectionPlain(t *testing.T) {
	ct := NewChatTemplate("")
	if ct.Type != ChatTemplateNone {
		t.Errorf("expected ChatTemplateNone, got %v", ct.Type)
	}
	if ct.StopToken != "" {
		t.Errorf("expected empty stop token, got %q", ct.StopToken)
	}

	ct2 := NewChatTemplate("some unknown template format")
	if ct2.Type != ChatTemplateNone {
		t.Errorf("expected ChatTemplateNone for unknown format, got %v", ct2.Type)
	}
}

func TestChatMLFormat(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateChatML, StopToken: "<|im_end|>"}
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}

	result := ct.Format(msgs)

	expected := "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
		"<|im_start|>user\nHello<|im_end|>\n" +
		"<|im_start|>assistant\n"

	if result != expected {
		t.Errorf("ChatML format mismatch.\nGot:\n%s\nExpected:\n%s", result, expected)
	}
}

func TestLlama3Format(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateLlama3, StopToken: "<|eot_id|>"}
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}

	result := ct.Format(msgs)

	expected := "<|begin_of_text|>" +
		"<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"

	if result != expected {
		t.Errorf("Llama3 format mismatch.\nGot:\n%s\nExpected:\n%s", result, expected)
	}
}

func TestPlainFallback(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateNone}
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}

	result := ct.Format(msgs)

	if !strings.Contains(result, "You are helpful.") {
		t.Error("plain format should contain system message")
	}
	if !strings.Contains(result, "Hello") {
		t.Error("plain format should contain user message")
	}
	if !strings.HasSuffix(result, "Assistant:") {
		t.Errorf("plain format should end with 'Assistant:', got %q", result[len(result)-20:])
	}
}

func TestMultiTurnFormat(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateChatML, StopToken: "<|im_end|>"}
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4"},
		{Role: "user", Content: "And 3+3?"},
	}

	result := ct.Format(msgs)

	if strings.Count(result, "<|im_start|>") != 5 {
		t.Errorf("expected 5 im_start tokens, got %d", strings.Count(result, "<|im_start|>"))
	}
	if strings.Count(result, "<|im_end|>") != 4 {
		t.Errorf("expected 4 im_end tokens, got %d", strings.Count(result, "<|im_end|>"))
	}
	if !strings.HasSuffix(result, "<|im_start|>assistant\n") {
		t.Error("should end with opening assistant turn")
	}
}

func TestFormatSimple(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateChatML, StopToken: "<|im_end|>"}

	result := ct.FormatSimple("You are helpful.", "Hello")

	if !strings.Contains(result, "<|im_start|>system\nYou are helpful.<|im_end|>") {
		t.Error("FormatSimple should include system message")
	}
	if !strings.Contains(result, "<|im_start|>user\nHello<|im_end|>") {
		t.Error("FormatSimple should include user message")
	}
}

func TestFormatSimpleNoSystem(t *testing.T) {
	ct := &ChatTemplate{Type: ChatTemplateChatML, StopToken: "<|im_end|>"}

	result := ct.FormatSimple("", "Hello")

	if strings.Contains(result, "system") {
		t.Error("FormatSimple with empty system should not include system role")
	}
	if !strings.Contains(result, "<|im_start|>user\nHello<|im_end|>") {
		t.Error("FormatSimple should include user message")
	}
}

func TestTemplateTypeString(t *testing.T) {
	tests := []struct {
		tt   ChatTemplateType
		want string
	}{
		{ChatTemplateChatML, "chatml"},
		{ChatTemplateLlama3, "llama3"},
		{ChatTemplateNone, "none"},
	}
	for _, tc := range tests {
		if got := tc.tt.String(); got != tc.want {
			t.Errorf("ChatTemplateType(%d).String() = %q, want %q", tc.tt, got, tc.want)
		}
	}
}
