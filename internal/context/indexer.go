package context

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	
	ignore "github.com/sabhiram/go-gitignore"
)

// FileEntry represents a file in the index
type FileEntry struct {
	Path         string    // Relative path from project root
	AbsPath      string    // Absolute path
	Size         int64     // File size in bytes
	Modified     time.Time // Last modification time
	Language     string    // Detected language
	IsTest       bool      // Whether this is a test file
	IsGenerated  bool      // Whether this is generated code
}

// FileIndex contains indexed project files
type FileIndex struct {
	Root     string                // Project root directory
	Files    map[string]*FileEntry // Path -> FileEntry
	Updated  time.Time             // When index was created
	FileCount int                  // Total files indexed
}

// Indexer indexes project files
type Indexer struct {
	root           string
	gitignore      *ignore.GitIgnore
	excludePatterns []string
	includePatterns []string
	maxFileSize    int64 // Skip files larger than this (bytes)
}

// IndexOptions configures the indexer
type IndexOptions struct {
	ExcludePatterns []string // Additional patterns to exclude
	IncludePatterns []string // Patterns to include (overrides excludes)
	MaxFileSize     int64    // Max file size to index (0 = no limit)
	FollowSymlinks  bool     // Whether to follow symlinks
}

// DefaultIndexOptions returns sensible defaults
func DefaultIndexOptions() IndexOptions {
	return IndexOptions{
		ExcludePatterns: []string{
			"*.log",
			"*.tmp",
			"*.swp",
			"*~",
			".DS_Store",
			"node_modules/",
			"vendor/",
			"__pycache__/",
			"*.pyc",
			".git/",
			".vscode/",
			".idea/",
			"dist/",
			"build/",
			"target/",
			"*.exe",
			"*.dll",
			"*.so",
			"*.dylib",
		},
		IncludePatterns: []string{
			"*.go",
			"*.py",
			"*.js",
			"*.ts",
			"*.tsx",
			"*.jsx",
			"*.java",
			"*.c",
			"*.cpp",
			"*.h",
			"*.hpp",
			"*.rs",
			"*.rb",
			"*.php",
			"*.cs",
			"*.swift",
			"*.kt",
			"*.scala",
			"*.md",
			"*.txt",
			"*.yaml",
			"*.yml",
			"*.json",
			"*.toml",
			"*.xml",
			"*.sh",
			"*.bash",
			"Makefile",
			"Dockerfile",
			"README",
			"LICENSE",
		},
		MaxFileSize:    10 * 1024 * 1024, // 10MB
		FollowSymlinks: false,
	}
}

// NewIndexer creates a new file indexer
func NewIndexer(root string, opts IndexOptions) (*Indexer, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}
	
	// Check if directory exists
	if _, err := os.Stat(absRoot); err != nil {
		return nil, fmt.Errorf("directory does not exist: %w", err)
	}
	
	indexer := &Indexer{
		root:            absRoot,
		excludePatterns: opts.ExcludePatterns,
		includePatterns: opts.IncludePatterns,
		maxFileSize:     opts.MaxFileSize,
	}
	
	// Load .gitignore if it exists
	gitignorePath := filepath.Join(absRoot, ".gitignore")
	if _, err := os.Stat(gitignorePath); err == nil {
		gi, err := ignore.CompileIgnoreFile(gitignorePath)
		if err != nil {
			return nil, fmt.Errorf("failed to parse .gitignore: %w", err)
		}
		indexer.gitignore = gi
	}
	
	return indexer, nil
}

// Index walks the directory tree and indexes all relevant files
func (idx *Indexer) Index() (*FileIndex, error) {
	index := &FileIndex{
		Root:    idx.root,
		Files:   make(map[string]*FileEntry),
		Updated: time.Now(),
	}
	
	err := filepath.Walk(idx.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files with errors
		}
		
		// Skip directories
		if info.IsDir() {
			// Check if directory should be excluded
			relPath, _ := filepath.Rel(idx.root, path)
			if idx.shouldExcludeDir(relPath) {
				return filepath.SkipDir
			}
			return nil
		}
		
		// Get relative path
		relPath, err := filepath.Rel(idx.root, path)
		if err != nil {
			return nil
		}
		
		// Check if file should be included
		if !idx.shouldIncludeFile(relPath, info) {
			return nil
		}
		
		// Create file entry
		entry := &FileEntry{
			Path:        relPath,
			AbsPath:     path,
			Size:        info.Size(),
			Modified:    info.ModTime(),
			Language:    detectLanguage(relPath),
			IsTest:      isTestFile(relPath),
			IsGenerated: isGeneratedFile(relPath),
		}
		
		index.Files[relPath] = entry
		index.FileCount++
		
		return nil
	})
	
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}
	
	return index, nil
}

// shouldExcludeDir checks if a directory should be excluded
func (idx *Indexer) shouldExcludeDir(relPath string) bool {
	// Check .gitignore
	if idx.gitignore != nil && idx.gitignore.MatchesPath(relPath) {
		return true
	}
	
	// Check exclude patterns
	for _, pattern := range idx.excludePatterns {
		if strings.HasSuffix(pattern, "/") {
			// Directory pattern
			if matched, _ := filepath.Match(strings.TrimSuffix(pattern, "/"), filepath.Base(relPath)); matched {
				return true
			}
		}
	}
	
	return false
}

// shouldIncludeFile checks if a file should be included in the index
func (idx *Indexer) shouldIncludeFile(relPath string, info os.FileInfo) bool {
	// Check file size
	if idx.maxFileSize > 0 && info.Size() > idx.maxFileSize {
		return false
	}
	
	// Check .gitignore
	if idx.gitignore != nil && idx.gitignore.MatchesPath(relPath) {
		return false
	}
	
	// Check exclude patterns
	for _, pattern := range idx.excludePatterns {
		if !strings.HasSuffix(pattern, "/") {
			if matched, _ := filepath.Match(pattern, filepath.Base(relPath)); matched {
				return false
			}
		}
	}
	
	// Check include patterns
	if len(idx.includePatterns) > 0 {
		for _, pattern := range idx.includePatterns {
			if matched, _ := filepath.Match(pattern, filepath.Base(relPath)); matched {
				return true
			}
		}
		return false // No include pattern matched
	}
	
	return true
}

// detectLanguage detects the programming language from file extension
func detectLanguage(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	
	languageMap := map[string]string{
		".go":         "Go",
		".py":         "Python",
		".js":         "JavaScript",
		".ts":         "TypeScript",
		".tsx":        "TypeScript",
		".jsx":        "JavaScript",
		".java":       "Java",
		".c":          "C",
		".cpp":        "C++",
		".cc":         "C++",
		".cxx":        "C++",
		".h":          "C/C++",
		".hpp":        "C++",
		".rs":         "Rust",
		".rb":         "Ruby",
		".php":        "PHP",
		".cs":         "C#",
		".swift":      "Swift",
		".kt":         "Kotlin",
		".scala":      "Scala",
		".sh":         "Shell",
		".bash":       "Bash",
		".zsh":        "Zsh",
		".md":         "Markdown",
		".yaml":       "YAML",
		".yml":        "YAML",
		".json":       "JSON",
		".toml":       "TOML",
		".xml":        "XML",
		".html":       "HTML",
		".css":        "CSS",
		".scss":       "SCSS",
		".sql":        "SQL",
	}
	
	if lang, ok := languageMap[ext]; ok {
		return lang
	}
	
	// Check for files without extension
	base := filepath.Base(path)
	if base == "Makefile" || base == "Dockerfile" || base == "Rakefile" {
		return "Configuration"
	}
	
	return "Unknown"
}

// isTestFile checks if a file is a test file
func isTestFile(path string) bool {
	base := filepath.Base(path)
	lower := strings.ToLower(base)
	
	// Common test patterns
	patterns := []string{
		"_test.go",
		"_test.py",
		".test.js",
		".test.ts",
		".spec.js",
		".spec.ts",
		"test_",
		"Test",
	}
	
	for _, pattern := range patterns {
		if strings.Contains(lower, strings.ToLower(pattern)) {
			return true
		}
	}
	
	return false
}

// isGeneratedFile checks if a file is likely generated code
func isGeneratedFile(path string) bool {
	base := filepath.Base(path)
	lower := strings.ToLower(base)
	
	// Common generated file patterns
	patterns := []string{
		".pb.go",        // Protocol buffers
		".pb.gw.go",     // GRPC gateway
		"_generated",
		".generated.",
		"gen_",
		"generated_",
	}
	
	for _, pattern := range patterns {
		if strings.Contains(lower, pattern) {
			return true
		}
	}
	
	return false
}

// GetFilesByLanguage returns files filtered by language
func (idx *FileIndex) GetFilesByLanguage(language string) []*FileEntry {
	var files []*FileEntry
	for _, entry := range idx.Files {
		if entry.Language == language {
			files = append(files, entry)
		}
	}
	return files
}

// GetFilesByPattern returns files matching a glob pattern
func (idx *FileIndex) GetFilesByPattern(pattern string) []*FileEntry {
	var files []*FileEntry
	for _, entry := range idx.Files {
		if matched, _ := filepath.Match(pattern, filepath.Base(entry.Path)); matched {
			files = append(files, entry)
		}
	}
	return files
}

// GetFile returns a file entry by path
func (idx *FileIndex) GetFile(path string) (*FileEntry, bool) {
	entry, ok := idx.Files[path]
	return entry, ok
}

// GetAllFiles returns all indexed files
func (idx *FileIndex) GetAllFiles() []*FileEntry {
	files := make([]*FileEntry, 0, len(idx.Files))
	for _, entry := range idx.Files {
		files = append(files, entry)
	}
	return files
}
