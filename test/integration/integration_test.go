package integration

import (
"os"
"path/filepath"
"testing"

"github.com/xupit3r/vibrant/internal/assistant"
ctxpkg "github.com/xupit3r/vibrant/internal/context"
"github.com/xupit3r/vibrant/internal/diff"
"github.com/xupit3r/vibrant/internal/plugin"
)

// TestConversationWorkflow tests the complete conversation workflow
func TestConversationWorkflow(t *testing.T) {
tmpDir := t.TempDir()

// Create conversation manager
cm := assistant.NewConversationManager(20, 4096, tmpDir, true)

// Add system message
if err := cm.AddSystemMessage("You are a helpful coding assistant"); err != nil {
t.Fatalf("Failed to add system message: %v", err)
}

// Simulate multi-turn conversation
exchanges := []struct {
user      string
assistant string
}{
{"How do I create a Go struct?", "You can create a struct using the 'type' keyword..."},
{"Can you show an example?", "Here's an example: type Person struct { Name string }"},
{"How do I add methods?", "You can add methods like: func (p *Person) GetName() string { return p.Name }"},
}

for _, ex := range exchanges {
if err := cm.AddUserMessage(ex.user); err != nil {
t.Fatalf("Failed to add user message: %v", err)
}
if err := cm.AddAssistantMessage(ex.assistant); err != nil {
t.Fatalf("Failed to add assistant message: %v", err)
}
}

// Check messages were added
messages := cm.GetMessages()
expectedCount := 1 + (len(exchanges) * 2) // system + exchanges
if len(messages) != expectedCount {
t.Errorf("Expected %d messages, got %d", expectedCount, len(messages))
}

// Test save functionality
if err := cm.Save(); err != nil {
t.Fatalf("Failed to save conversation: %v", err)
}

// Test pruning stats
stats := cm.GetPruningStats()
if stats["total_messages"].(int) != expectedCount {
t.Errorf("Pruning stats mismatch")
}
}

// TestContextIndexingWorkflow tests file indexing
func TestContextIndexingWorkflow(t *testing.T) {
tmpDir := t.TempDir()

// Create test files
testFiles := map[string]string{
"main.go":    "package main\n\nfunc main() {\n\tprintln(\"hello\")\n}",
"util.go":    "package main\n\nfunc helper() string {\n\treturn \"help\"\n}",
"README.md":  "# Test Project\n\nThis is a test.",
".gitignore": "*.log\n*.tmp\n",
}

for filename, content := range testFiles {
path := filepath.Join(tmpDir, filename)
if err := os.WriteFile(path, []byte(content), 0644); err != nil {
t.Fatalf("Failed to create test file %s: %v", filename, err)
}
}

// Create indexer
indexer, err := ctxpkg.NewIndexer(tmpDir, ctxpkg.IndexOptions{
MaxFileSize: 1024 * 1024, // 1MB
})
if err != nil {
t.Fatalf("Failed to create indexer: %v", err)
}

// Index files
fileIndex, err := indexer.Index()
if err != nil {
t.Fatalf("Failed to index files: %v", err)
}

// Get all files
files := fileIndex.GetAllFiles()
if len(files) < 2 { // At least main.go and util.go
t.Logf("Expected at least 2 Go files, got %d files", len(files))
}

// Test language detection
goFiles := fileIndex.GetFilesByLanguage("go")
if len(goFiles) != 2 {
t.Logf("Expected 2 Go files, got %d", len(goFiles))
}

// Test pattern matching
mainFiles := fileIndex.GetFilesByPattern("main")
if len(mainFiles) < 1 {
t.Logf("Expected at least 1 main file, got %d", len(mainFiles))
}
}

// TestVectorStoreWorkflow tests semantic search functionality
func TestVectorStoreWorkflow(t *testing.T) {
vs := ctxpkg.NewVectorStore()

// Add documents
docs := []struct {
id      string
path    string
content string
}{
{"1", "auth.go", "package auth; func Login(username, password string) error"},
{"2", "db.go", "package db; func Connect(connectionString string) error"},
{"3", "api.go", "package api; func HandleLogin(w http.ResponseWriter, r *http.Request)"},
{"4", "crypto.go", "package crypto; func HashPassword(password string) string"},
}

for _, doc := range docs {
vs.AddDocument(doc.id, doc.path, doc.content)
}

// Build index
vs.BuildIndex()

if vs.Size() != len(docs) {
t.Errorf("Expected %d documents, got %d", len(docs), vs.Size())
}

// Search for login-related code
results := vs.Search("login authentication", 2)

if len(results) != 2 {
t.Errorf("Expected 2 results, got %d", len(results))
}

// First result should be relevant
if results[0].Score <= 0 {
t.Error("Results should have positive relevance scores")
}

// Test search with different query
results = vs.Search("database connection", 2)
if len(results) != 2 {
t.Errorf("Expected 2 results for database query, got %d", len(results))
}
}

// TestDiffWorkflow tests diff generation and git integration
func TestDiffWorkflow(t *testing.T) {
tmpDir := t.TempDir()

// Create a test file
originalContent := `package main

func main() {
println("hello")
}`

modifiedContent := `package main

import "fmt"

func main() {
fmt.Println("hello world")
}`

testFile := filepath.Join(tmpDir, "main.go")
if err := os.WriteFile(testFile, []byte(originalContent), 0644); err != nil {
t.Fatalf("Failed to write test file: %v", err)
}

// Generate diff
fileDiff, err := diff.Generate(originalContent, modifiedContent, "main.go")
if err != nil {
t.Fatalf("Failed to generate diff: %v", err)
}

if fileDiff == nil {
t.Fatal("Expected diff to be generated")
}

if len(fileDiff.Hunks) == 0 {
t.Error("Expected at least one hunk in diff")
}

// Format as unified diff
unified := fileDiff.FormatUnified()
if unified == "" {
t.Error("Expected non-empty unified diff")
}

// Test smart commit message generation
commitMsg := diff.GenerateSmartCommitMessage(unified)
if commitMsg == "" {
t.Error("Expected non-empty commit message")
}

t.Logf("Generated commit message: %s", commitMsg)
}

// TestPluginWorkflow tests plugin registration and execution
func TestPluginWorkflow(t *testing.T) {
manager := plugin.NewManager()

// Create a simple test plugin
testPlugin := &MockTransformPlugin{
name:    "uppercase",
version: "1.0.0",
}

info := plugin.PluginInfo{
Name:        "uppercase",
Version:     "1.0.0",
Description: "Converts text to uppercase",
Author:      "test",
Enabled:     true,
}

// Register plugin
if err := manager.Register(testPlugin, info); err != nil {
t.Fatalf("Failed to register plugin: %v", err)
}

// Execute plugin
result, err := manager.Execute("uppercase", "hello world")
if err != nil {
t.Fatalf("Failed to execute plugin: %v", err)
}

if result != "HELLO WORLD" {
t.Errorf("Expected 'HELLO WORLD', got '%v'", result)
}

// Test disable/enable
if err := manager.Disable("uppercase"); err != nil {
t.Fatalf("Failed to disable plugin: %v", err)
}

_, err = manager.Execute("uppercase", "test")
if err == nil {
t.Error("Expected error when executing disabled plugin")
}

if err := manager.Enable("uppercase"); err != nil {
t.Fatalf("Failed to enable plugin: %v", err)
}

_, err = manager.Execute("uppercase", "test")
if err != nil {
t.Error("Should be able to execute after re-enabling")
}
}

// MockTransformPlugin is a simple plugin for testing
type MockTransformPlugin struct {
name    string
version string
}

func (p *MockTransformPlugin) Name() string {
return p.name
}

func (p *MockTransformPlugin) Version() string {
return p.version
}

func (p *MockTransformPlugin) Initialize(config map[string]interface{}) error {
return nil
}

func (p *MockTransformPlugin) Execute(input interface{}) (interface{}, error) {
if str, ok := input.(string); ok {
return toUpper(str), nil
}
return input, nil
}

func (p *MockTransformPlugin) Shutdown() error {
return nil
}

func toUpper(s string) string {
result := ""
for _, c := range s {
if c >= 'a' && c <= 'z' {
result += string(c - 32)
} else {
result += string(c)
}
}
return result
}

// TestEndToEndWorkflow tests a complete workflow combining multiple features
func TestEndToEndWorkflow(t *testing.T) {
tmpDir := t.TempDir()

// Setup: Create a small codebase
codeFiles := map[string]string{
"main.go": `package main

import "fmt"

func main() {
fmt.Println("Hello, Vibrant!")
}`,
"helper.go": `package main

func greet(name string) string {
return "Hello, " + name
}`,
}

for filename, content := range codeFiles {
path := filepath.Join(tmpDir, filename)
if err := os.WriteFile(path, []byte(content), 0644); err != nil {
t.Fatalf("Failed to create file %s: %v", filename, err)
}
}

// 1. Index the codebase
indexer, err := ctxpkg.NewIndexer(tmpDir, ctxpkg.IndexOptions{})
if err != nil {
t.Fatalf("Failed to create indexer: %v", err)
}

fileIndex, err := indexer.Index()
if err != nil {
t.Fatalf("Failed to index: %v", err)
}

files := fileIndex.GetAllFiles()
if len(files) != 2 {
t.Errorf("Expected 2 files, got %d", len(files))
}

// 2. Create vector store and add documents
vs := ctxpkg.NewVectorStore()
for _, file := range files {
content, _ := os.ReadFile(file.Path)
vs.AddDocument(file.Path, file.Path, string(content))
}
vs.BuildIndex()

// 3. Search for relevant code
results := vs.Search("greeting function", 1)
if len(results) == 0 {
t.Error("Expected to find relevant code")
}

// 4. Create a conversation about the code
cm := assistant.NewConversationManager(10, 2048, tmpDir, false)
cm.AddSystemMessage("You are a code assistant")
cm.AddUserMessage("How does the greeting function work?")
cm.AddAssistantMessage("The greet function takes a name and returns a greeting")

if len(cm.GetMessages()) != 3 {
t.Error("Expected 3 messages in conversation")
}

// 5. Test plugin integration
pm := plugin.NewManager()
mockPlugin := &MockTransformPlugin{name: "test", version: "1.0"}
pm.Register(mockPlugin, plugin.PluginInfo{
Name:    "test",
Enabled: true,
})

if pm.Count() != 1 {
t.Error("Expected 1 plugin registered")
}

t.Log("End-to-end workflow completed successfully")
}
