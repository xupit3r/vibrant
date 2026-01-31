package assistant

import (
	"testing"
	"time"
)

func TestNewConversationManager(t *testing.T) {
	tmpDir := t.TempDir()
	
	cm := NewConversationManager(20, 4096, tmpDir, false)
	
	if cm.conversation.MaxHistory != 20 {
		t.Errorf("Expected max history 20, got %d", cm.conversation.MaxHistory)
	}
	
	if cm.conversation.ContextWindow != 4096 {
		t.Errorf("Expected context window 4096, got %d", cm.conversation.ContextWindow)
	}
	
	if len(cm.conversation.Messages) != 0 {
		t.Errorf("Expected empty message list, got %d messages", len(cm.conversation.Messages))
	}
}

func TestAddMessage(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	err := cm.AddUserMessage("Hello, world!")
	if err != nil {
		t.Fatalf("Failed to add user message: %v", err)
	}
	
	if len(cm.GetMessages()) != 1 {
		t.Errorf("Expected 1 message, got %d", len(cm.GetMessages()))
	}
	
	msg := cm.GetMessages()[0]
	if msg.Role != "user" {
		t.Errorf("Expected role 'user', got '%s'", msg.Role)
	}
	
	if msg.Content != "Hello, world!" {
		t.Errorf("Expected content 'Hello, world!', got '%s'", msg.Content)
	}
	
	if msg.Tokens <= 0 {
		t.Error("Expected positive token count")
	}
}

func TestAddMultipleMessages(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	messages := []struct {
		role    string
		content string
	}{
		{"user", "What is Go?"},
		{"assistant", "Go is a programming language."},
		{"user", "Tell me more"},
		{"assistant", "Go was created at Google."},
	}
	
	for _, msg := range messages {
		cm.AddMessage(msg.role, msg.content)
	}
	
	if len(cm.GetMessages()) != len(messages) {
		t.Errorf("Expected %d messages, got %d", len(messages), len(cm.GetMessages()))
	}
	
	// Verify order preserved
	for i, msg := range cm.GetMessages() {
		if msg.Role != messages[i].role {
			t.Errorf("Message %d: expected role '%s', got '%s'", 
				i, messages[i].role, msg.Role)
		}
	}
}

func TestGetRecentMessages(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	// Add 10 messages
	for i := 0; i < 10; i++ {
		cm.AddUserMessage("Message " + string(rune('0'+i)))
	}
	
	// Get recent 3
	recent := cm.GetRecentMessages(3)
	if len(recent) != 3 {
		t.Errorf("Expected 3 recent messages, got %d", len(recent))
	}
	
	// Should be the last 3 messages
	if recent[0].Content != "Message 7" {
		t.Errorf("Expected 'Message 7', got '%s'", recent[0].Content)
	}
	
	// Get more than available
	all := cm.GetRecentMessages(100)
	if len(all) != 10 {
		t.Errorf("Expected all 10 messages, got %d", len(all))
	}
}

func TestConversationPruning(t *testing.T) {
	cm := NewConversationManager(5, 10000, "", false)
	
	// Add more messages than max history
	for i := 0; i < 10; i++ {
		cm.AddUserMessage("Message " + string(rune('0'+i)))
	}
	
	// Should only keep last 5
	messages := cm.GetMessages()
	if len(messages) > 5 {
		t.Errorf("Expected at most 5 messages after pruning, got %d", len(messages))
	}
}

func TestClearConversation(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	cm.AddUserMessage("Test message 1")
	cm.AddUserMessage("Test message 2")
	
	if len(cm.GetMessages()) != 2 {
		t.Fatal("Messages not added correctly")
	}
	
	cm.Clear()
	
	if len(cm.GetMessages()) != 0 {
		t.Errorf("Expected empty conversation after clear, got %d messages", 
			len(cm.GetMessages()))
	}
	
	if cm.GetTokenCount() != 0 {
		t.Errorf("Expected 0 tokens after clear, got %d", cm.GetTokenCount())
	}
}

func TestGetTokenCount(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	initialTokens := cm.GetTokenCount()
	if initialTokens != 0 {
		t.Errorf("Expected 0 initial tokens, got %d", initialTokens)
	}
	
	cm.AddUserMessage("This is a test message with some words")
	
	tokens := cm.GetTokenCount()
	if tokens <= 0 {
		t.Error("Expected positive token count after adding message")
	}
	
	cm.AddAssistantMessage("This is a response")
	
	newTokens := cm.GetTokenCount()
	if newTokens <= tokens {
		t.Error("Token count should increase after adding another message")
	}
}

func TestMessageTimestamps(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	before := time.Now()
	cm.AddUserMessage("Test")
	after := time.Now()
	
	msg := cm.GetMessages()[0]
	
	if msg.Timestamp.Before(before) || msg.Timestamp.After(after) {
		t.Error("Message timestamp not set correctly")
	}
}

func TestSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	
	// Create and populate conversation
	cm1 := NewConversationManager(20, 4096, tmpDir, false)
	cm1.AddUserMessage("Hello")
	cm1.AddAssistantMessage("Hi there")
	
	conversationID := cm1.conversation.ID
	
	// Save
	err := cm1.Save()
	if err != nil {
		t.Fatalf("Failed to save conversation: %v", err)
	}
	
	// Load in new manager
	cm2 := NewConversationManager(20, 4096, tmpDir, false)
	err = cm2.Load(conversationID)
	if err != nil {
		t.Fatalf("Failed to load conversation: %v", err)
	}
	
	// Verify messages restored
	if len(cm2.GetMessages()) != 2 {
		t.Errorf("Expected 2 messages after load, got %d", len(cm2.GetMessages()))
	}
	
	if cm2.GetMessages()[0].Content != "Hello" {
		t.Error("First message content not restored correctly")
	}
	
	if cm2.GetMessages()[1].Content != "Hi there" {
		t.Error("Second message content not restored correctly")
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		text     string
		minTokens int
		maxTokens int
	}{
		{"", 0, 0},
		{"Hello", 1, 2},
		{"This is a longer test message", 5, 10},
		{"A very long message with many words that should result in more tokens", 10, 20},
	}
	
	for _, tt := range tests {
		tokens := estimateTokens(tt.text)
		if tokens < tt.minTokens || tokens > tt.maxTokens {
			t.Errorf("estimateTokens(%q) = %d; want between %d and %d", 
				tt.text, tokens, tt.minTokens, tt.maxTokens)
		}
	}
}
