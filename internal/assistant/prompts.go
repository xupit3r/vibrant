package assistant

import (
	"fmt"
	"strings"
)

// PromptTemplate defines a prompt template
type PromptTemplate struct {
	Name         string // Template name
	System       string // System prompt
	UserPrefix   string // Prefix for user messages
	AssistPrefix string // Prefix for assistant messages
	ContextFmt   string // Format string for context
}

// DefaultTemplate returns the default coding assistant template
var DefaultTemplate = PromptTemplate{
	Name:   "default",
	System: `You are Vibrant, an expert coding assistant that helps developers with programming questions, code review, debugging, and architectural decisions.

Guidelines:
- Provide concise, accurate, and practical answers
- Include code examples when relevant
- Explain your reasoning clearly
- Consider edge cases and best practices
- Be honest if you're unsure about something`,
	UserPrefix:   "User",
	AssistPrefix: "Assistant",
	ContextFmt:   "<context>\n%s\n</context>\n\n",
}

// ConciseTemplate for brief responses
var ConciseTemplate = PromptTemplate{
	Name:   "concise",
	System: `You are Vibrant, a coding assistant. Provide brief, direct answers to coding questions. Be concise but accurate.`,
	UserPrefix:   "Q",
	AssistPrefix: "A",
	ContextFmt:   "Context:\n%s\n\n",
}

// DetailedTemplate for in-depth explanations
var DetailedTemplate = PromptTemplate{
	Name:   "detailed",
	System: `You are Vibrant, an expert coding mentor. Provide comprehensive, detailed explanations for coding questions. Include:
- Clear explanations of concepts
- Multiple approaches when applicable
- Code examples with comments
- Trade-offs and considerations
- Best practices and common pitfalls`,
	UserPrefix:   "Question",
	AssistPrefix: "Detailed Answer",
	ContextFmt:   "<project_context>\n%s\n</project_context>\n\n",
}

// DebugTemplate for debugging assistance
var DebugTemplate = PromptTemplate{
	Name:   "debug",
	System: `You are Vibrant, a debugging expert. Help identify and fix bugs by:
- Analyzing error messages and stack traces
- Identifying root causes
- Suggesting fixes with explanations
- Recommending preventive measures
- Considering edge cases`,
	UserPrefix:   "Issue",
	AssistPrefix: "Analysis",
	ContextFmt:   "<code_context>\n%s\n</code_context>\n\n",
}

// ReviewTemplate for code review
var ReviewTemplate = PromptTemplate{
	Name:   "review",
	System: `You are Vibrant, a code reviewer. Review code for:
- Correctness and logic errors
- Code quality and readability
- Performance considerations
- Security issues
- Best practices and idioms
Provide constructive feedback with specific suggestions.`,
	UserPrefix:   "Code to Review",
	AssistPrefix: "Review",
	ContextFmt:   "<code>\n%s\n</code>\n\n",
}

// GetTemplate returns a template by name
func GetTemplate(name string) PromptTemplate {
	switch name {
	case "concise":
		return ConciseTemplate
	case "detailed":
		return DetailedTemplate
	case "debug":
		return DebugTemplate
	case "review":
		return ReviewTemplate
	default:
		return DefaultTemplate
	}
}

// ListTemplates returns all available template names
func ListTemplates() []string {
	return []string{"default", "concise", "detailed", "debug", "review"}
}

// PromptBuilder builds prompts from templates
type PromptBuilder struct {
	template PromptTemplate
}

// NewPromptBuilder creates a new prompt builder
func NewPromptBuilder(templateName string) *PromptBuilder {
	return &PromptBuilder{
		template: GetTemplate(templateName),
	}
}

// BuildPrompt builds a complete prompt
func (pb *PromptBuilder) BuildPrompt(userQuery string, context string, conversationHistory string) string {
	var sb strings.Builder
	
	// Add system prompt
	sb.WriteString(pb.template.System)
	sb.WriteString("\n\n")
	
	// Add context if provided
	if context != "" {
		sb.WriteString(fmt.Sprintf(pb.template.ContextFmt, context))
	}
	
	// Add conversation history if provided
	if conversationHistory != "" {
		sb.WriteString("Conversation History:\n")
		sb.WriteString(conversationHistory)
		sb.WriteString("\n")
	}
	
	// Add user query
	sb.WriteString(fmt.Sprintf("%s: %s\n\n", pb.template.UserPrefix, userQuery))
	sb.WriteString(fmt.Sprintf("%s:", pb.template.AssistPrefix))
	
	return sb.String()
}

// BuildSimplePrompt builds a simple prompt without context or history
func (pb *PromptBuilder) BuildSimplePrompt(userQuery string) string {
	return pb.BuildPrompt(userQuery, "", "")
}

// BuildWithContext builds a prompt with context
func (pb *PromptBuilder) BuildWithContext(userQuery string, context string) string {
	return pb.BuildPrompt(userQuery, context, "")
}

// BuildWithHistory builds a prompt with conversation history
func (pb *PromptBuilder) BuildWithHistory(userQuery string, conversationHistory string) string {
	return pb.BuildPrompt(userQuery, "", conversationHistory)
}
