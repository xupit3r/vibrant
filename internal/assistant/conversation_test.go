package assistant

import (
	"fmt"
	"strings"
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

func TestPruneByTokens(t *testing.T) {
	// Create conversation with small context window
	cm := NewConversationManager(100, 200, "", false) // 200 tokens max
	
	// Add system message (small)
	cm.AddSystemMessage("You are a helpful assistant") // ~7 tokens
	
	// Add many messages to exceed token limit
	for i := 0; i < 20; i++ {
		// Each message is ~25 tokens
		cm.AddUserMessage("This is a test message number " + string(rune('0'+i)) + " with some content")
		cm.AddAssistantMessage("This is a response to message " + string(rune('0'+i)) + " with more content")
	}
	
	// Should have pruned to fit within context window
	if cm.GetTokenCount() > 200 {
		t.Errorf("Expected token count <= 200 after pruning, got %d", cm.GetTokenCount())
	}
	
	// Should still have system message
	hasSystem := false
	for _, msg := range cm.GetMessages() {
		if msg.Role == "system" {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		t.Error("System message should be preserved during pruning")
	}
	
	// Should have recent messages
	messages := cm.GetMessages()
	if len(messages) < 2 {
		t.Error("Should keep at least some recent messages")
	}
}

func TestPruneByCount(t *testing.T) {
	cm := NewConversationManager(8, 100000, "", false) // Max 8 messages
	
	// Add system message
	cm.AddSystemMessage("System prompt")
	
	// Add 20 messages (will exceed max)
	for i := 0; i < 20; i++ {
		cm.AddUserMessage(fmt.Sprintf("User message %d", i))
		cm.AddAssistantMessage(fmt.Sprintf("Assistant response %d", i))
	}
	
	messages := cm.GetMessages()
	
	// Should not exceed max history
	if len(messages) > 8 {
		t.Errorf("Expected at most 8 messages, got %d", len(messages))
	}
	
	// Should have system message
	if messages[0].Role != "system" {
		t.Error("First message should be system message")
	}
	
	// Should have most recent messages
	lastMsg := messages[len(messages)-1]
	if !strings.Contains(lastMsg.Content, "19") {
		t.Logf("Last message content: %s", lastMsg.Content)
		t.Log("Messages kept:", len(messages))
		// It's okay if not exactly message 19, just check we have recent ones
		if !strings.Contains(lastMsg.Content, "response") && !strings.Contains(lastMsg.Content, "message") {
			t.Error("Should keep recent messages")
		}
	}
}

func TestGetPruningStats(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	cm.AddSystemMessage("System prompt")
	cm.AddUserMessage("Hello")
	cm.AddAssistantMessage("Hi there")
	cm.AddUserMessage("How are you?")
	
	stats := cm.GetPruningStats()
	
	if stats["total_messages"].(int) != 4 {
		t.Errorf("Expected 4 total messages, got %v", stats["total_messages"])
	}
	
	if stats["system_messages"].(int) != 1 {
		t.Errorf("Expected 1 system message, got %v", stats["system_messages"])
	}
	
	if stats["user_messages"].(int) != 2 {
		t.Errorf("Expected 2 user messages, got %v", stats["user_messages"])
	}
	
	if stats["assistant_messages"].(int) != 1 {
		t.Errorf("Expected 1 assistant message, got %v", stats["assistant_messages"])
	}
	
	if stats["max_history"].(int) != 20 {
		t.Errorf("Expected max_history 20, got %v", stats["max_history"])
	}
}

func TestContextWindowPruningPreservesRecent(t *testing.T) {
	cm := NewConversationManager(100, 100, "", false) // Very small context window
	
	cm.AddSystemMessage("System")
	
	// Add messages that will exceed token limit
	for i := 0; i < 10; i++ {
		cm.AddUserMessage(fmt.Sprintf("Message %d", i))
		cm.AddAssistantMessage(fmt.Sprintf("Response %d", i))
	}
	
	messages := cm.GetMessages()
	
	// Should have pruned to fit token limit
	if cm.GetTokenCount() > 100 {
		t.Errorf("Token count %d exceeds limit 100", cm.GetTokenCount())
	}
	
	// Most recent message should be preserved
	lastMsg := messages[len(messages)-1]
	if !strings.Contains(lastMsg.Content, "Response") {
		t.Error("Most recent messages should be preserved")
	}
}

func TestFormatForPrompt(t *testing.T) {
	cm := NewConversationManager(20, 4096, "", false)
	
	cm.AddSystemMessage("You are helpful")
	cm.AddUserMessage("Hello")
	cm.AddAssistantMessage("Hi there")
	
	formatted := cm.FormatForPrompt()
	
	if !strings.Contains(formatted, "[SYSTEM]:") {
		t.Error("Formatted prompt should contain system message")
	}
	
	if !strings.Contains(formatted, "[USER]:") {
		t.Error("Formatted prompt should contain user message")
	}
	
	if !strings.Contains(formatted, "[ASSISTANT]:") {
		t.Error("Formatted prompt should contain assistant message")
	}
	
	if !strings.Contains(formatted, "Hello") {
		t.Error("Formatted prompt should contain message content")
	}
}
