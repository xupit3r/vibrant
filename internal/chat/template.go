package chat

import (
	"strings"
)

// Message represents a chat message with a role and content.
type Message struct {
	Role    string
	Content string
}

// ChatTemplateType identifies the chat template format
type ChatTemplateType int

const (
	ChatTemplateNone   ChatTemplateType = iota // Plain text (base models)
	ChatTemplateChatML                         // ChatML format (Qwen, Yi)
	ChatTemplateLlama3                         // Llama 3/3.1 format
)

// String returns the name of the template type
func (t ChatTemplateType) String() string {
	switch t {
	case ChatTemplateChatML:
		return "chatml"
	case ChatTemplateLlama3:
		return "llama3"
	default:
		return "none"
	}
}

// ChatTemplate formats messages for a specific model's expected input format
type ChatTemplate struct {
	Type      ChatTemplateType
	StopToken string // e.g. "<|im_end|>" or "<|eot_id|>"
}

// NewChatTemplate auto-detects the chat template format from the raw GGUF template string.
// If rawTemplate is empty, returns a plain-text fallback template.
func NewChatTemplate(rawTemplate string) *ChatTemplate {
	if strings.Contains(rawTemplate, "<|im_start|>") {
		return &ChatTemplate{Type: ChatTemplateChatML, StopToken: "<|im_end|>"}
	}
	if strings.Contains(rawTemplate, "<|start_header_id|>") {
		return &ChatTemplate{Type: ChatTemplateLlama3, StopToken: "<|eot_id|>"}
	}
	return &ChatTemplate{Type: ChatTemplateNone}
}

// Format converts a slice of messages to the model-specific prompt string.
// The final message should be the last user message; an assistant turn is opened
// at the end so the model can generate a completion.
func (ct *ChatTemplate) Format(messages []Message) string {
	switch ct.Type {
	case ChatTemplateChatML:
		return ct.formatChatML(messages)
	case ChatTemplateLlama3:
		return ct.formatLlama3(messages)
	default:
		return ct.formatPlain(messages)
	}
}

// FormatSimple is a convenience method that builds a system+user prompt.
func (ct *ChatTemplate) FormatSimple(system, user string) string {
	var msgs []Message
	if system != "" {
		msgs = append(msgs, Message{Role: "system", Content: system})
	}
	msgs = append(msgs, Message{Role: "user", Content: user})
	return ct.Format(msgs)
}

func (ct *ChatTemplate) formatChatML(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(m.Role)
		sb.WriteByte('\n')
		sb.WriteString(m.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func (ct *ChatTemplate) formatLlama3(messages []Message) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, m := range messages {
		sb.WriteString("<|start_header_id|>")
		sb.WriteString(m.Role)
		sb.WriteString("<|end_header_id|>\n\n")
		sb.WriteString(m.Content)
		sb.WriteString("<|eot_id|>")
	}
	sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return sb.String()
}

func (ct *ChatTemplate) formatPlain(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		switch m.Role {
		case "system":
			sb.WriteString(m.Content)
			sb.WriteString("\n\n")
		case "user":
			sb.WriteString("User: ")
			sb.WriteString(m.Content)
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString("Assistant: ")
			sb.WriteString(m.Content)
			sb.WriteString("\n\n")
		}
	}
	sb.WriteString("Assistant:")
	return sb.String()
}
