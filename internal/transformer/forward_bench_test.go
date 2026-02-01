package transformer

import (
"os"
"testing"

"github.com/xupit3r/vibrant/internal/gguf"
)

func BenchmarkForwardPass(b *testing.B) {
// Load real model for benchmarking
modelPath := os.Getenv("HOME") + "/.vibrant/models/qwen2.5-coder-7b-q4.gguf"

ggufFile, err := gguf.ParseGGUF(modelPath)
if err != nil {
b.Skipf("Cannot load model: %v", err)
}

model, err := NewModel(ggufFile)
if err != nil {
b.Fatalf("Failed to create model: %v", err)
}

// Simple input: single token
tokens := [][]int{{1, 100, 200}}  // BOS + 2 tokens

b.ResetTimer()
b.ReportAllocs()

for i := 0; i < b.N; i++ {
_, err := model.Forward(tokens, true)
if err != nil {
b.Fatalf("Forward failed: %v", err)
}
}
}
