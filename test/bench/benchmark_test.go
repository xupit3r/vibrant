package bench

import (
"testing"

"github.com/xupit3r/vibrant/internal/assistant"
ctxpkg "github.com/xupit3r/vibrant/internal/context"
"github.com/xupit3r/vibrant/internal/diff"
)

// BenchmarkConversationAdd benchmarks adding messages to conversation
func BenchmarkConversationAdd(b *testing.B) {
cm := assistant.NewConversationManager(1000, 100000, "", false)

b.ResetTimer()
for i := 0; i < b.N; i++ {
cm.AddUserMessage("Test message")
}
}

// BenchmarkConversationPruning benchmarks pruning with many messages
func BenchmarkConversationPruning(b *testing.B) {
for i := 0; i < b.N; i++ {
cm := assistant.NewConversationManager(50, 10000, "", false)

// Add many messages to trigger pruning
for j := 0; j < 100; j++ {
cm.AddUserMessage("Message number " + string(rune('0'+j%10)))
cm.AddAssistantMessage("Response number " + string(rune('0'+j%10)))
}
}
}

// BenchmarkVectorStoreAdd benchmarks adding documents to vector store
func BenchmarkVectorStoreAdd(b *testing.B) {
vs := ctxpkg.NewVectorStore()
content := "package main\n\nfunc TestFunction() {\n\treturn\n}"

b.ResetTimer()
for i := 0; i < b.N; i++ {
vs.AddDocument(string(rune('a'+i%26)), "file.go", content)
}
}

// BenchmarkVectorStoreSearch benchmarks searching the vector store
func BenchmarkVectorStoreSearch(b *testing.B) {
vs := ctxpkg.NewVectorStore()

// Populate with sample documents
for i := 0; i < 100; i++ {
vs.AddDocument(
string(rune('a'+i%26)),
"file.go",
"package main func test authentication login user password",
)
}
vs.BuildIndex()

b.ResetTimer()
for i := 0; i < b.N; i++ {
vs.Search("authentication login", 5)
}
}

// BenchmarkDiffGenerate benchmarks diff generation
func BenchmarkDiffGenerate(b *testing.B) {
oldContent := `package main

func main() {
println("hello")
println("world")
println("test")
}`

newContent := `package main

import "fmt"

func main() {
fmt.Println("hello")
fmt.Println("world")
fmt.Println("test")
fmt.Println("new line")
}`

b.ResetTimer()
for i := 0; i < b.N; i++ {
diff.Generate(oldContent, newContent, "main.go")
}
}

// BenchmarkSmartCommitMessage benchmarks commit message generation
func BenchmarkSmartCommitMessage(b *testing.B) {
diffContent := `diff --git a/main.go b/main.go
--- a/main.go
+++ b/main.go
@@ -1,5 +1,7 @@
 package main
 
+import "fmt"
+
 func main() {
-println("hello")
+fmt.Println("hello world")
 }`

b.ResetTimer()
for i := 0; i < b.N; i++ {
diff.GenerateSmartCommitMessage(diffContent)
}
}
