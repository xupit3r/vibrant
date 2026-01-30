# Assistant Core Specification

## Overview
The assistant core coordinates between the model, context system, and user interface.

## Conversation Management

### Conversation State
```go
type ConversationManager struct {
    history    []Message
    maxHistory int
    contextWindow int
}

type Message struct {
    Role      string    // "user" or "assistant"
    Content   string
    Timestamp time.Time
    Tokens    int
}
```

### History Management
- **Storage**: In-memory during session
- **Persistence**: Optional save to ~/.vibrant/sessions/
- **Pruning**: Remove oldest messages when approaching context limit

## Prompt System

### Template Structure
```go
type PromptTemplate struct {
    System       string
    UserPrefix   string
    AssistPrefix string
    ContextFmt   string
}

var CodingTemplate = PromptTemplate{
    System: "You are Vibrant, a helpful coding assistant...",
    UserPrefix: "<|user|>",
    AssistPrefix: "<|assistant|>",
    ContextFmt: "<context>\n%s\n</context>\n\n",
}
```

### Context Injection
```
[System Prompt]
[Context Files]
[Conversation History]
[Current Query]
```

## Status
- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Implementation**: Phase 5
