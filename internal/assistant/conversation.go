package assistant

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Message represents a message in the conversation
type Message struct {
	Role      string    `json:"role"`      // "user" or "assistant" or "system"
	Content   string    `json:"content"`   // Message content
	Timestamp time.Time `json:"timestamp"` // When message was created
	Tokens    int       `json:"tokens"`    // Estimated token count
}

// Conversation manages a conversation history
type Conversation struct {
	ID            string    `json:"id"`             // Unique conversation ID
	Messages      []Message `json:"messages"`       // Conversation messages
	Created       time.Time `json:"created"`        // When conversation started
	LastUpdated   time.Time `json:"last_updated"`   // Last message time
	MaxHistory    int       `json:"max_history"`    // Max messages to keep
	ContextWindow int       `json:"context_window"` // Model context window size
	TotalTokens   int       `json:"total_tokens"`   // Total tokens in conversation
}

// ConversationManager manages conversation state
type ConversationManager struct {
	conversation  *Conversation
	saveDir       string
	autoSave      bool
}

// NewConversationManager creates a new conversation manager
func NewConversationManager(maxHistory int, contextWindow int, saveDir string, autoSave bool) *ConversationManager {
	// Expand home directory
	if len(saveDir) > 0 && saveDir[0] == '~' {
		home, _ := os.UserHomeDir()
		saveDir = filepath.Join(home, saveDir[1:])
	}
	
	// Create save directory
	if saveDir != "" {
		os.MkdirAll(saveDir, 0755)
	}
	
	return &ConversationManager{
		conversation: &Conversation{
			ID:            generateID(),
			Messages:      []Message{},
			Created:       time.Now(),
			LastUpdated:   time.Now(),
			MaxHistory:    maxHistory,
			ContextWindow: contextWindow,
			TotalTokens:   0,
		},
		saveDir:  saveDir,
		autoSave: autoSave,
	}
}

// AddMessage adds a message to the conversation
func (cm *ConversationManager) AddMessage(role string, content string) error {
	// Estimate tokens
	tokens := estimateTokens(content)
	
	message := Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Tokens:    tokens,
	}
	
	cm.conversation.Messages = append(cm.conversation.Messages, message)
	cm.conversation.LastUpdated = time.Now()
	cm.conversation.TotalTokens += tokens
	
	// Prune if needed
	if err := cm.pruneIfNeeded(); err != nil {
		return err
	}
	
	// Auto-save if enabled
	if cm.autoSave && cm.saveDir != "" {
		return cm.Save()
	}
	
	return nil
}

// AddSystemMessage adds a system message
func (cm *ConversationManager) AddSystemMessage(content string) error {
	return cm.AddMessage("system", content)
}

// AddUserMessage adds a user message
func (cm *ConversationManager) AddUserMessage(content string) error {
	return cm.AddMessage("user", content)
}

// AddAssistantMessage adds an assistant message
func (cm *ConversationManager) AddAssistantMessage(content string) error {
	return cm.AddMessage("assistant", content)
}

// GetMessages returns all messages in the conversation
func (cm *ConversationManager) GetMessages() []Message {
	return cm.conversation.Messages
}

// GetRecentMessages returns the N most recent messages
func (cm *ConversationManager) GetRecentMessages(n int) []Message {
	if n >= len(cm.conversation.Messages) {
		return cm.conversation.Messages
	}
	return cm.conversation.Messages[len(cm.conversation.Messages)-n:]
}

// GetTokenCount returns the total token count
func (cm *ConversationManager) GetTokenCount() int {
	return cm.conversation.TotalTokens
}

// Clear clears the conversation history
func (cm *ConversationManager) Clear() {
	cm.conversation.Messages = []Message{}
	cm.conversation.TotalTokens = 0
	cm.conversation.LastUpdated = time.Now()
}

// pruneIfNeeded prunes old messages if limits are exceeded
func (cm *ConversationManager) pruneIfNeeded() error {
	// Check message count limit
	if cm.conversation.MaxHistory > 0 && len(cm.conversation.Messages) > cm.conversation.MaxHistory {
		// Remove oldest messages (but keep system messages)
		var pruned []Message
		systemMessages := []Message{}
		otherMessages := []Message{}
		
		for _, msg := range cm.conversation.Messages {
			if msg.Role == "system" {
				systemMessages = append(systemMessages, msg)
			} else {
				otherMessages = append(otherMessages, msg)
			}
		}
		
		// Keep system messages + most recent other messages
		keepCount := cm.conversation.MaxHistory - len(systemMessages)
		if keepCount < 0 {
			keepCount = 0
		}
		
		if len(otherMessages) > keepCount {
			otherMessages = otherMessages[len(otherMessages)-keepCount:]
		}
		
		pruned = append(systemMessages, otherMessages...)
		cm.conversation.Messages = pruned
		
		// Recalculate total tokens
		cm.recalculateTokens()
	}
	
	// Check context window limit
	if cm.conversation.ContextWindow > 0 && cm.conversation.TotalTokens > cm.conversation.ContextWindow {
		// Remove messages from the middle (keep system and recent messages)
		for cm.conversation.TotalTokens > cm.conversation.ContextWindow && len(cm.conversation.Messages) > 2 {
			// Find first non-system message to remove
			for i, msg := range cm.conversation.Messages {
				if msg.Role != "system" {
					// Remove this message
					cm.conversation.Messages = append(cm.conversation.Messages[:i], cm.conversation.Messages[i+1:]...)
					cm.conversation.TotalTokens -= msg.Tokens
					break
				}
			}
		}
	}
	
	return nil
}

// recalculateTokens recalculates total token count
func (cm *ConversationManager) recalculateTokens() {
	total := 0
	for _, msg := range cm.conversation.Messages {
		total += msg.Tokens
	}
	cm.conversation.TotalTokens = total
}

// Save saves the conversation to disk
func (cm *ConversationManager) Save() error {
	if cm.saveDir == "" {
		return fmt.Errorf("save directory not configured")
	}
	
	filename := filepath.Join(cm.saveDir, fmt.Sprintf("%s.json", cm.conversation.ID))
	
	data, err := json.MarshalIndent(cm.conversation, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal conversation: %w", err)
	}
	
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write conversation: %w", err)
	}
	
	return nil
}

// Load loads a conversation from disk
func (cm *ConversationManager) Load(conversationID string) error {
	if cm.saveDir == "" {
		return fmt.Errorf("save directory not configured")
	}
	
	filename := filepath.Join(cm.saveDir, fmt.Sprintf("%s.json", conversationID))
	
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read conversation: %w", err)
	}
	
	var conv Conversation
	if err := json.Unmarshal(data, &conv); err != nil {
		return fmt.Errorf("failed to unmarshal conversation: %w", err)
	}
	
	cm.conversation = &conv
	
	return nil
}

// ListSaved lists all saved conversations
func ListSavedConversations(saveDir string) ([]string, error) {
	// Expand home directory
	if len(saveDir) > 0 && saveDir[0] == '~' {
		home, _ := os.UserHomeDir()
		saveDir = filepath.Join(home, saveDir[1:])
	}
	
	entries, err := os.ReadDir(saveDir)
	if err != nil {
		return nil, err
	}
	
	var conversations []string
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".json" {
			conversations = append(conversations, entry.Name()[:len(entry.Name())-5])
		}
	}
	
	return conversations, nil
}

// estimateTokens estimates token count for text
func estimateTokens(text string) int {
	// Rough estimate: ~4 characters per token
	return len(text) / 4
}

// generateID generates a unique conversation ID
func generateID() string {
	return fmt.Sprintf("conv_%d", time.Now().UnixNano())
}

// FormatForPrompt formats conversation history for inclusion in a prompt
func (cm *ConversationManager) FormatForPrompt() string {
	var result string
	for _, msg := range cm.conversation.Messages {
		switch msg.Role {
		case "system":
			result += fmt.Sprintf("[SYSTEM]: %s\n\n", msg.Content)
		case "user":
			result += fmt.Sprintf("[USER]: %s\n\n", msg.Content)
		case "assistant":
			result += fmt.Sprintf("[ASSISTANT]: %s\n\n", msg.Content)
		}
	}
	return result
}
