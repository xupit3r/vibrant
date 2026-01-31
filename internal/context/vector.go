package context

import (
	"math"
	"sort"
	"strings"
)

// VectorStore provides semantic search capabilities for code
type VectorStore struct {
	documents []Document
	idf       map[string]float64
	indexed   bool
}

// Document represents a code document with its vector representation
type Document struct {
	ID      string
	Path    string
	Content string
	Tokens  []string
	TF      map[string]float64
	Score   float64
}

// NewVectorStore creates a new vector store
func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents: []Document{},
		idf:       make(map[string]float64),
		indexed:   false,
	}
}

// AddDocument adds a document to the store
func (vs *VectorStore) AddDocument(id, path, content string) {
	tokens := tokenize(content)
	tf := computeTF(tokens)
	
	doc := Document{
		ID:      id,
		Path:    path,
		Content: content,
		Tokens:  tokens,
		TF:      tf,
	}
	
	vs.documents = append(vs.documents, doc)
	vs.indexed = false
}

// BuildIndex builds the IDF index for all documents
func (vs *VectorStore) BuildIndex() {
	if vs.indexed {
		return
	}
	
	// Count document frequency for each term
	df := make(map[string]int)
	for _, doc := range vs.documents {
		seen := make(map[string]bool)
		for _, token := range doc.Tokens {
			if !seen[token] {
				df[token]++
				seen[token] = true
			}
		}
	}
	
	// Compute IDF for each term
	numDocs := float64(len(vs.documents))
	for term, count := range df {
		vs.idf[term] = math.Log(numDocs / float64(count))
	}
	
	vs.indexed = true
}

// Search finds the most relevant documents for a query
func (vs *VectorStore) Search(query string, topK int) []Document {
	if !vs.indexed {
		vs.BuildIndex()
	}
	
	if len(vs.documents) == 0 {
		return []Document{}
	}
	
	// Tokenize query and compute TF
	queryTokens := tokenize(query)
	queryTF := computeTF(queryTokens)
	
	// Compute TF-IDF for query
	queryVector := make(map[string]float64)
	for term, tf := range queryTF {
		if idf, ok := vs.idf[term]; ok {
			queryVector[term] = tf * idf
		}
	}
	
	// Score each document
	results := make([]Document, len(vs.documents))
	copy(results, vs.documents)
	
	for i := range results {
		results[i].Score = cosineSimilarity(queryVector, results[i].TF, vs.idf)
	}
	
	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	
	// Return top K
	if topK > len(results) {
		topK = len(results)
	}
	
	return results[:topK]
}

// Clear clears all documents from the store
func (vs *VectorStore) Clear() {
	vs.documents = []Document{}
	vs.idf = make(map[string]float64)
	vs.indexed = false
}

// Size returns the number of documents in the store
func (vs *VectorStore) Size() int {
	return len(vs.documents)
}

// tokenize splits text into tokens
func tokenize(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Split on non-alphanumeric characters
	var tokens []string
	var current strings.Builder
	
	for _, char := range text {
		if (char >= 'a' && char <= 'z') || (char >= '0' && char <= '9') || char == '_' {
			current.WriteRune(char)
		} else {
			if current.Len() > 0 {
				token := current.String()
				if len(token) > 2 { // Filter short tokens
					tokens = append(tokens, token)
				}
				current.Reset()
			}
		}
	}
	
	// Add last token
	if current.Len() > 0 {
		token := current.String()
		if len(token) > 2 {
			tokens = append(tokens, token)
		}
	}
	
	return tokens
}

// computeTF computes term frequency for a list of tokens
func computeTF(tokens []string) map[string]float64 {
	tf := make(map[string]float64)
	total := float64(len(tokens))
	
	if total == 0 {
		return tf
	}
	
	// Count occurrences
	counts := make(map[string]int)
	for _, token := range tokens {
		counts[token]++
	}
	
	// Compute normalized frequency
	for token, count := range counts {
		tf[token] = float64(count) / total
	}
	
	return tf
}

// cosineSimilarity computes cosine similarity between query and document vectors
func cosineSimilarity(query map[string]float64, docTF map[string]float64, idf map[string]float64) float64 {
	var dotProduct, queryMag, docMag float64
	
	// Compute document TF-IDF vector magnitude and dot product
	for term, tf := range docTF {
		if idfVal, ok := idf[term]; ok {
			docTFIDF := tf * idfVal
			docMag += docTFIDF * docTFIDF
			
			if queryTFIDF, ok := query[term]; ok {
				dotProduct += queryTFIDF * docTFIDF
			}
		}
	}
	
	// Compute query vector magnitude
	for _, val := range query {
		queryMag += val * val
	}
	
	// Compute cosine similarity
	if queryMag == 0 || docMag == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(queryMag) * math.Sqrt(docMag))
}
