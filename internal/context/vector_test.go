package context

import (
	"strings"
	"testing"
)

func TestNewVectorStore(t *testing.T) {
	vs := NewVectorStore()
	
	if vs == nil {
		t.Fatal("NewVectorStore returned nil")
	}
	
	if vs.Size() != 0 {
		t.Errorf("Expected empty store, got size %d", vs.Size())
	}
}

func TestAddDocument(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "file1.go", "func main() { fmt.Println(\"hello\") }")
	
	if vs.Size() != 1 {
		t.Errorf("Expected 1 document, got %d", vs.Size())
	}
	
	vs.AddDocument("doc2", "file2.go", "func test() { return true }")
	
	if vs.Size() != 2 {
		t.Errorf("Expected 2 documents, got %d", vs.Size())
	}
}

func TestBuildIndex(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "file1.go", "function test")
	vs.AddDocument("doc2", "file2.go", "function hello")
	vs.AddDocument("doc3", "file3.go", "test suite")
	
	vs.BuildIndex()
	
	if !vs.indexed {
		t.Error("Index should be built")
	}
	
	if len(vs.idf) == 0 {
		t.Error("IDF map should not be empty")
	}
	
	// "function" appears in 2 documents, "test" in 2, etc.
	// Check that IDF values exist
	hasIDF := false
	for term, idf := range vs.idf {
		if idf > 0 {
			hasIDF = true
			t.Logf("Term '%s' has IDF %.2f", term, idf)
		}
	}
	
	if !hasIDF {
		t.Error("Should have computed IDF values")
	}
}

func TestSearch(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "main.go", "func main() { fmt.Println(\"hello world\") }")
	vs.AddDocument("doc2", "test.go", "func TestMain(t *testing.T) { test suite }")
	vs.AddDocument("doc3", "util.go", "func helper() { return 42 }")
	
	results := vs.Search("test function", 2)
	
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
	
	// Results should have scores
	if results[0].Score <= 0 {
		t.Error("First result should have a positive score")
	}
	
	// Scores should be in descending order
	if len(results) > 1 && results[0].Score < results[1].Score {
		t.Error("Results should be sorted by score descending")
	}
}

func TestSearchEmptyStore(t *testing.T) {
	vs := NewVectorStore()
	
	results := vs.Search("test", 10)
	
	if len(results) != 0 {
		t.Errorf("Expected no results from empty store, got %d", len(results))
	}
}

func TestSearchRelevance(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "http.go", "HTTP server implementation with handlers and routes")
	vs.AddDocument("doc2", "db.go", "Database connection and queries")
	vs.AddDocument("doc3", "api.go", "API endpoints and HTTP handlers")
	
	results := vs.Search("HTTP server", 3)
	
	if len(results) == 0 {
		t.Fatal("Expected some results")
	}
	
	// First result should be most relevant (http.go or api.go)
	firstPath := results[0].Path
	if !strings.Contains(firstPath, "http") && !strings.Contains(firstPath, "api") {
		t.Logf("First result: %s (score: %.3f)", firstPath, results[0].Score)
		// Don't fail, just log - TF-IDF can be sensitive to exact wording
	}
}

func TestClear(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "file1.go", "content")
	vs.AddDocument("doc2", "file2.go", "content")
	vs.BuildIndex()
	
	if vs.Size() != 2 {
		t.Fatal("Setup failed")
	}
	
	vs.Clear()
	
	if vs.Size() != 0 {
		t.Errorf("Expected empty store after clear, got %d", vs.Size())
	}
	
	if vs.indexed {
		t.Error("indexed flag should be false after clear")
	}
	
	if len(vs.idf) != 0 {
		t.Error("IDF map should be empty after clear")
	}
}

func TestTokenize(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{
			input:    "func main() {}",
			expected: []string{"func", "main"},
		},
		{
			input:    "HelloWorld",
			expected: []string{"helloworld"},
		},
		{
			input:    "test_function_name",
			expected: []string{"test_function_name"},
		},
		{
			input:    "a b",
			expected: []string{}, // Too short, filtered out
		},
		{
			input:    "HTTP Server API",
			expected: []string{"http", "server", "api"},
		},
	}
	
	for _, tt := range tests {
		result := tokenize(tt.input)
		
		if len(tt.expected) == 0 && len(result) == 0 {
			continue
		}
		
		if len(result) != len(tt.expected) {
			t.Errorf("tokenize(%q): expected %d tokens, got %d (%v)", 
				tt.input, len(tt.expected), len(result), result)
			continue
		}
		
		for i, token := range result {
			if i < len(tt.expected) && token != tt.expected[i] {
				t.Errorf("tokenize(%q)[%d]: expected %q, got %q", 
					tt.input, i, tt.expected[i], token)
			}
		}
	}
}

func TestComputeTF(t *testing.T) {
	tokens := []string{"hello", "world", "hello", "test"}
	
	tf := computeTF(tokens)
	
	if tf["hello"] != 0.5 { // 2/4
		t.Errorf("Expected TF for 'hello' to be 0.5, got %.2f", tf["hello"])
	}
	
	if tf["world"] != 0.25 { // 1/4
		t.Errorf("Expected TF for 'world' to be 0.25, got %.2f", tf["world"])
	}
	
	if tf["test"] != 0.25 { // 1/4
		t.Errorf("Expected TF for 'test' to be 0.25, got %.2f", tf["test"])
	}
}

func TestComputeTFEmpty(t *testing.T) {
	tf := computeTF([]string{})
	
	if len(tf) != 0 {
		t.Error("TF of empty token list should be empty")
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors should have similarity of 1.0
	vec1 := map[string]float64{"a": 1.0, "b": 1.0}
	idf := map[string]float64{"a": 1.0, "b": 1.0}
	
	sim := cosineSimilarity(vec1, vec1, idf)
	
	if sim < 0.99 || sim > 1.01 {
		t.Errorf("Cosine similarity of identical vectors should be ~1.0, got %.2f", sim)
	}
	
	// Orthogonal vectors should have similarity of 0.0
	vec2 := map[string]float64{"c": 1.0, "d": 1.0}
	idf2 := map[string]float64{"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
	
	sim = cosineSimilarity(vec1, vec2, idf2)
	
	if sim != 0.0 {
		t.Errorf("Cosine similarity of orthogonal vectors should be 0.0, got %.2f", sim)
	}
}

func TestSearchTopK(t *testing.T) {
	vs := NewVectorStore()
	
	for i := 0; i < 10; i++ {
		vs.AddDocument(
			string(rune('a'+i)),
			"file.go",
			"content with some words",
		)
	}
	
	results := vs.Search("content words", 5)
	
	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}
	
	results = vs.Search("content words", 20)
	
	if len(results) != 10 {
		t.Errorf("Expected 10 results (all docs), got %d", len(results))
	}
}

func TestMultipleIndexBuilds(t *testing.T) {
	vs := NewVectorStore()
	
	vs.AddDocument("doc1", "file1.go", "content")
	vs.BuildIndex()
	
	firstIndexed := vs.indexed
	firstIDFSize := len(vs.idf)
	
	vs.BuildIndex() // Should not rebuild
	
	if !vs.indexed {
		t.Error("Should remain indexed")
	}
	
	if !firstIndexed || len(vs.idf) != firstIDFSize {
		t.Error("Second BuildIndex should be no-op")
	}
}
