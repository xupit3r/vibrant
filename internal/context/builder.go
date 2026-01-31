package context

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Context represents the code context for a query
type Context struct {
	Files       []ContextFile // Files included in context
	Summary     string        // Project structure summary
	TokenCount  int           // Estimated token count
	ProjectRoot string        // Project root directory
}

// ContextFile represents a file included in the context
type ContextFile struct {
	Path      string  // Relative path
	Content   string  // File content
	Language  string  // Programming language
	Relevance float64 // Relevance score (0.0-1.0)
	Lines     int     // Number of lines
}

// Builder builds context for queries
type Builder struct {
	index      *FileIndex
	maxTokens  int
	maxFiles   int
	strategy   string
}

// BuilderOptions configures the context builder
type BuilderOptions struct {
	MaxTokens int    // Maximum tokens to include
	MaxFiles  int    // Maximum files to include
	Strategy  string // Context building strategy (smart, full, minimal)
}

// DefaultBuilderOptions returns sensible defaults
func DefaultBuilderOptions() BuilderOptions {
	return BuilderOptions{
		MaxTokens: 3000,
		MaxFiles:  50,
		Strategy:  "smart",
	}
}

// NewBuilder creates a new context builder
func NewBuilder(index *FileIndex, opts BuilderOptions) *Builder {
	return &Builder{
		index:     index,
		maxTokens: opts.MaxTokens,
		maxFiles:  opts.MaxFiles,
		strategy:  opts.Strategy,
	}
}

// Build builds context for a query
func (b *Builder) Build(query string) (*Context, error) {
	switch b.strategy {
	case "minimal":
		return b.buildMinimal()
	case "full":
		return b.buildFull(query)
	case "smart":
		return b.buildSmart(query)
	default:
		return b.buildSmart(query)
	}
}

// buildMinimal builds minimal context (README only)
func (b *Builder) buildMinimal() (*Context, error) {
	ctx := &Context{
		Files:       []ContextFile{},
		ProjectRoot: b.index.Root,
	}
	
	// Find README
	for _, entry := range b.index.Files {
		if strings.HasPrefix(strings.ToUpper(filepath.Base(entry.Path)), "README") {
			content, err := os.ReadFile(entry.AbsPath)
			if err == nil {
				ctx.Files = append(ctx.Files, ContextFile{
					Path:      entry.Path,
					Content:   string(content),
					Language:  entry.Language,
					Relevance: 1.0,
					Lines:     countLines(string(content)),
				})
				ctx.TokenCount = estimateTokens(string(content))
				break
			}
		}
	}
	
	ctx.Summary = b.buildSummary()
	
	return ctx, nil
}

// buildFull builds full context (all files up to limit)
func (b *Builder) buildFull(query string) (*Context, error) {
	ctx := &Context{
		Files:       []ContextFile{},
		ProjectRoot: b.index.Root,
	}
	
	// Get all files sorted by relevance
	files := b.rankFilesByRelevance(query)
	
	tokenCount := 0
	for i, entry := range files {
		if i >= b.maxFiles {
			break
		}
		
		content, err := os.ReadFile(entry.AbsPath)
		if err != nil {
			continue
		}
		
		contentStr := string(content)
		tokens := estimateTokens(contentStr)
		
		if tokenCount+tokens > b.maxTokens {
			break
		}
		
		ctx.Files = append(ctx.Files, ContextFile{
			Path:      entry.Path,
			Content:   contentStr,
			Language:  entry.Language,
			Relevance: scoreRelevance(query, entry),
			Lines:     countLines(contentStr),
		})
		
		tokenCount += tokens
	}
	
	ctx.TokenCount = tokenCount
	ctx.Summary = b.buildSummary()
	
	return ctx, nil
}

// buildSmart builds smart context (most relevant files)
func (b *Builder) buildSmart(query string) (*Context, error) {
	ctx := &Context{
		Files:       []ContextFile{},
		ProjectRoot: b.index.Root,
	}
	
	// Always include README if it exists
	tokenCount := 0
	for _, entry := range b.index.Files {
		if strings.HasPrefix(strings.ToUpper(filepath.Base(entry.Path)), "README") {
			content, err := os.ReadFile(entry.AbsPath)
			if err == nil {
				contentStr := string(content)
				tokens := estimateTokens(contentStr)
				if tokens < b.maxTokens/4 { // README shouldn't use more than 25% of budget
					ctx.Files = append(ctx.Files, ContextFile{
						Path:      entry.Path,
						Content:   contentStr,
						Language:  entry.Language,
						Relevance: 1.0,
						Lines:     countLines(contentStr),
					})
					tokenCount += tokens
				}
			}
			break
		}
	}
	
	// Get ranked files
	files := b.rankFilesByRelevance(query)
	
	// Add most relevant files
	for i, entry := range files {
		if i >= b.maxFiles {
			break
		}
		
		// Skip if already added (README)
		alreadyAdded := false
		for _, cf := range ctx.Files {
			if cf.Path == entry.Path {
				alreadyAdded = true
				break
			}
		}
		if alreadyAdded {
			continue
		}
		
		content, err := os.ReadFile(entry.AbsPath)
		if err != nil {
			continue
		}
		
		contentStr := string(content)
		tokens := estimateTokens(contentStr)
		
		if tokenCount+tokens > b.maxTokens {
			break
		}
		
		ctx.Files = append(ctx.Files, ContextFile{
			Path:      entry.Path,
			Content:   contentStr,
			Language:  entry.Language,
			Relevance: scoreRelevance(query, entry),
			Lines:     countLines(contentStr),
		})
		
		tokenCount += tokens
	}
	
	ctx.TokenCount = tokenCount
	ctx.Summary = b.buildSummary()
	
	return ctx, nil
}

// rankFilesByRelevance ranks files by relevance to query
func (b *Builder) rankFilesByRelevance(query string) []*FileEntry {
	queryLower := strings.ToLower(query)
	queryWords := strings.Fields(queryLower)
	
	// Score all files
	type scoredFile struct {
		entry *FileEntry
		score float64
	}
	
	var scored []scoredFile
	for _, entry := range b.index.Files {
		// Skip test files unless query mentions tests
		if entry.IsTest && !containsAny(queryLower, []string{"test", "testing", "spec"}) {
			continue
		}
		
		// Skip generated files
		if entry.IsGenerated {
			continue
		}
		
		score := scoreRelevance(query, entry)
		
		// Boost score for certain file types
		if strings.HasSuffix(entry.Path, "main.go") || strings.HasSuffix(entry.Path, "__init__.py") {
			score *= 1.5
		}
		
		// Boost for files with query words in path
		for _, word := range queryWords {
			if strings.Contains(strings.ToLower(entry.Path), word) {
				score *= 1.3
			}
		}
		
		scored = append(scored, scoredFile{entry: entry, score: score})
	}
	
	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})
	
	// Return sorted entries
	result := make([]*FileEntry, len(scored))
	for i, sf := range scored {
		result[i] = sf.entry
	}
	
	return result
}

// scoreRelevance scores a file's relevance to a query
func scoreRelevance(query string, entry *FileEntry) float64 {
	score := 0.5 // Base score
	
	queryLower := strings.ToLower(query)
	pathLower := strings.ToLower(entry.Path)
	
	// Check if query keywords appear in path
	words := strings.Fields(queryLower)
	for _, word := range words {
		if strings.Contains(pathLower, word) {
			score += 0.3
		}
	}
	
	// Boost for certain languages based on query
	if strings.Contains(queryLower, "go") && entry.Language == "Go" {
		score += 0.4
	} else if strings.Contains(queryLower, "python") && entry.Language == "Python" {
		score += 0.4
	} else if strings.Contains(queryLower, "javascript") && entry.Language == "JavaScript" {
		score += 0.4
	}
	
	// Boost for main/entry point files
	if strings.Contains(pathLower, "main") || strings.Contains(pathLower, "index") {
		score += 0.2
	}
	
	// Penalize very large files
	if entry.Size > 100*1024 { // > 100KB
		score *= 0.8
	}
	
	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}
	
	return score
}

// buildSummary builds a project structure summary
func (b *Builder) buildSummary() string {
	var sb strings.Builder
	
	sb.WriteString("Project Structure:\n")
	
	// Group by language
	langCount := make(map[string]int)
	for _, entry := range b.index.Files {
		if !entry.IsTest && !entry.IsGenerated {
			langCount[entry.Language]++
		}
	}
	
	// Sort languages by count
	type langStat struct {
		lang  string
		count int
	}
	var langs []langStat
	for lang, count := range langCount {
		langs = append(langs, langStat{lang, count})
	}
	sort.Slice(langs, func(i, j int) bool {
		return langs[i].count > langs[j].count
	})
	
	sb.WriteString("Languages:\n")
	for _, ls := range langs {
		if ls.count > 0 {
			sb.WriteString(fmt.Sprintf("  - %s: %d files\n", ls.lang, ls.count))
		}
	}
	
	sb.WriteString(fmt.Sprintf("\nTotal files indexed: %d\n", b.index.FileCount))
	
	return sb.String()
}

// estimateTokens estimates token count for text
func estimateTokens(text string) int {
	// Rough estimate: ~4 characters per token for code
	return len(text) / 4
}

// countLines counts lines in text
func countLines(text string) int {
	return strings.Count(text, "\n") + 1
}

// containsAny checks if s contains any of the substrings
func containsAny(s string, substrs []string) bool {
	for _, substr := range substrs {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

// FormatContext formats context for inclusion in a prompt
func (c *Context) FormatContext() string {
	var sb strings.Builder
	
	sb.WriteString("<context>\n\n")
	
	// Add summary
	sb.WriteString(c.Summary)
	sb.WriteString("\n")
	
	// Add files
	if len(c.Files) > 0 {
		sb.WriteString("Relevant files:\n\n")
		for _, file := range c.Files {
			sb.WriteString(fmt.Sprintf("--- %s (%s) ---\n", file.Path, file.Language))
			sb.WriteString(file.Content)
			sb.WriteString("\n\n")
		}
	}
	
	sb.WriteString("</context>")
	
	return sb.String()
}
