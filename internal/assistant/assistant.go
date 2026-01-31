package assistant

import (
	"context"
	"fmt"
	
	ctxpkg "github.com/xupit3r/vibrant/internal/context"
	"github.com/xupit3r/vibrant/internal/llm"
	"github.com/xupit3r/vibrant/internal/model"
)

// Assistant orchestrates the overall assistant functionality
type Assistant struct {
	modelManager *model.Manager
	llmManager   *llm.Manager
	conversation *ConversationManager
	promptBuilder *PromptBuilder
	contextIndex  *ctxpkg.FileIndex
}

// AssistantConfig configures the assistant
type AssistantConfig struct {
	ModelID          string
	TemplateName     string
	MaxHistory       int
	ContextWindow    int
	SaveDir          string
	AutoSave         bool
	MaxContextTokens int
	ContextStrategy  string
}

// DefaultAssistantConfig returns sensible defaults
func DefaultAssistantConfig() AssistantConfig {
	return AssistantConfig{
		ModelID:          "auto",
		TemplateName:     "default",
		MaxHistory:       20,
		ContextWindow:    4096,
		SaveDir:          "~/.vibrant/sessions",
		AutoSave:         true,
		MaxContextTokens: 3000,
		ContextStrategy:  "smart",
	}
}

// NewAssistant creates a new assistant
func NewAssistant(modelManager *model.Manager, config AssistantConfig) (*Assistant, error) {
	// Initialize LLM manager
	llmManager := llm.NewManager(modelManager)
	
	// Load model
	if err := llmManager.LoadModel(config.ModelID); err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	
	// Create conversation manager
	conversation := NewConversationManager(
		config.MaxHistory,
		config.ContextWindow,
		config.SaveDir,
		config.AutoSave,
	)
	
	// Add system message
	template := GetTemplate(config.TemplateName)
	conversation.AddSystemMessage(template.System)
	
	// Create prompt builder
	promptBuilder := NewPromptBuilder(config.TemplateName)
	
	return &Assistant{
		modelManager:  modelManager,
		llmManager:    llmManager,
		conversation:  conversation,
		promptBuilder: promptBuilder,
		contextIndex:  nil,
	}, nil
}

// SetContextIndex sets the context index for the assistant
func (a *Assistant) SetContextIndex(index *ctxpkg.FileIndex) {
	a.contextIndex = index
}

// Ask asks a question and returns a response
func (a *Assistant) Ask(ctx context.Context, query string, opts llm.GenerateOptions) (string, error) {
	// Add user message to history
	if err := a.conversation.AddUserMessage(query); err != nil {
		return "", fmt.Errorf("failed to add user message: %w", err)
	}
	
	// Build context string if index is available
	var contextStr string
	if a.contextIndex != nil {
		builder := ctxpkg.NewBuilder(a.contextIndex, ctxpkg.BuilderOptions{
			MaxTokens: 3000,
			MaxFiles:  50,
			Strategy:  "smart",
		})
		
		codeContext, err := builder.Build(query)
		if err == nil && len(codeContext.Files) > 0 {
			contextStr = codeContext.FormatContext()
		}
	}
	
	// Get conversation history (last 5 exchanges)
	recentMessages := a.conversation.GetRecentMessages(10)
	var historyStr string
	for _, msg := range recentMessages {
		if msg.Role != "system" {
			historyStr += fmt.Sprintf("[%s]: %s\n\n", msg.Role, msg.Content)
		}
	}
	
	// Build prompt
	prompt := a.promptBuilder.BuildPrompt(query, contextStr, historyStr)
	
	// Generate response
	response, err := a.llmManager.Generate(ctx, prompt, opts)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	
	// Add assistant message to history
	if err := a.conversation.AddAssistantMessage(response); err != nil {
		return "", fmt.Errorf("failed to add assistant message: %w", err)
	}
	
	return response, nil
}

// AskStream asks a question and returns a streaming response
func (a *Assistant) AskStream(ctx context.Context, query string, opts llm.GenerateOptions) (<-chan string, error) {
	// Add user message to history
	if err := a.conversation.AddUserMessage(query); err != nil {
		return nil, fmt.Errorf("failed to add user message: %w", err)
	}
	
	// Build context string if index is available
	var contextStr string
	if a.contextIndex != nil {
		builder := ctxpkg.NewBuilder(a.contextIndex, ctxpkg.BuilderOptions{
			MaxTokens: 3000,
			MaxFiles:  50,
			Strategy:  "smart",
		})
		
		codeContext, err := builder.Build(query)
		if err == nil && len(codeContext.Files) > 0 {
			contextStr = codeContext.FormatContext()
		}
	}
	
	// Get conversation history
	recentMessages := a.conversation.GetRecentMessages(10)
	var historyStr string
	for _, msg := range recentMessages {
		if msg.Role != "system" {
			historyStr += fmt.Sprintf("[%s]: %s\n\n", msg.Role, msg.Content)
		}
	}
	
	// Build prompt
	prompt := a.promptBuilder.BuildPrompt(query, contextStr, historyStr)
	
	// Generate streaming response
	stream, err := a.llmManager.GenerateStream(ctx, prompt, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}
	
	// Collect response in background to add to history
	responseCh := make(chan string, 10)
	go func() {
		var fullResponse strings.Builder
		for token := range stream {
			fullResponse.WriteString(token)
			responseCh <- token
		}
		close(responseCh)
		
		// Add to history
		a.conversation.AddAssistantMessage(fullResponse.String())
	}()
	
	return responseCh, nil
}

// ClearHistory clears the conversation history
func (a *Assistant) ClearHistory() {
	a.conversation.Clear()
	
	// Re-add system message
	template := GetTemplate(a.promptBuilder.template.Name)
	a.conversation.AddSystemMessage(template.System)
}

// GetHistory returns the conversation history
func (a *Assistant) GetHistory() []Message {
	return a.conversation.GetMessages()
}

// SaveConversation saves the current conversation
func (a *Assistant) SaveConversation() error {
	return a.conversation.Save()
}

// LoadConversation loads a saved conversation
func (a *Assistant) LoadConversation(conversationID string) error {
	return a.conversation.Load(conversationID)
}

// Close closes the assistant and releases resources
func (a *Assistant) Close() error {
	return a.llmManager.Close()
}
