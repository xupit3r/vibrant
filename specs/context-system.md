# Context System Specification

## Overview
The context system indexes project files and retrieves relevant code context for the assistant.

## File Indexing

### Directory Walker
- **Library**: github.com/karrick/godirwalk
- **Concurrency**: Goroutine pool (NumCPU workers)
- **Filtering**: .gitignore aware + custom excludes

### Index Structure
```go
type FileIndex struct {
    Files     map[string]*FileEntry
    Updated   time.Time
    ProjectRoot string
}

type FileEntry struct {
    Path         string
    Size         int64
    Modified     time.Time
    Language     string
    Relevance    float64
}
```

## Context Retrieval

### Strategies
1. **Smart** (default): Keyword matching + file type + recency
2. **Full**: Include all relevant files up to token limit
3. **Minimal**: README + current file only

### Context Builder
```go
type ContextBuilder interface {
    Build(query string, maxTokens int) (*Context, error)
}

type Context struct {
    Files     []ContextFile
    Summary   string
    Tokens    int
}

type ContextFile struct {
    Path      string
    Content   string
    Relevance float64
}
```

## Status
- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Implementation**: Phase 4
