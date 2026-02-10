# Tokenizer Specification

## Overview

The `internal/tokenizer/` package implements BPE (Byte-Pair Encoding) tokenization for LLM models. It provides encoding (text → token IDs) and decoding (token IDs → text) functionality with support for special tokens.

## Design Goals

1. **GGUF Integration**: Load vocabulary and merge rules from GGUF metadata
2. **BPE Algorithm**: Implement standard byte-level BPE tokenization
3. **Special Tokens**: Support for BOS, EOS, PAD, UNK tokens
4. **Performance**: Fast encoding/decoding with minimal allocations
5. **Compatibility**: Match llama.cpp tokenization behavior

## Architecture

```
┌─────────────────────────────────────┐
│          Tokenizer                  │
│                                     │
│  ┌──────────┐  ┌───────────┐       │
│  │  Vocab   │  │  Merges   │       │
│  │ (string  │  │ (BPE      │       │
│  │  → ID)   │  │  rules)   │       │
│  └──────────┘  └───────────┘       │
│                                     │
│  ┌──────────────────────┐           │
│  │  Encode(text) → []ID │           │
│  │  Decode([]ID) → text │           │
│  └──────────────────────┘           │
└─────────────────────────────────────┘
         ▲
         │ Load from
         │
┌────────┴─────────────────────────────┐
│       GGUF Metadata                  │
│  - tokenizer.ggml.tokens             │
│  - tokenizer.ggml.scores             │
│  - tokenizer.ggml.merges             │
│  - tokenizer.ggml.bos_token_id       │
│  - tokenizer.ggml.eos_token_id       │
└──────────────────────────────────────┘
```

## Data Structures

### Tokenizer

```go
type Tokenizer struct {
    // Vocabulary: token string → token ID
    vocab map[string]int

    // Reverse vocabulary: token ID → token string
    tokens []string

    // Token scores (used for ranking in BPE merges)
    scores []float32

    // BPE merge rules: "token1 token2" → merge rank
    // Lower rank = applied earlier (higher priority)
    merges map[string]int

    // Special token IDs
    bosID int // Beginning of sequence
    eosID int // End of sequence
    padID int // Padding token
    unkID int // Unknown token

    // Model type (e.g., "gpt2", "llama")
    modelType string
}
```

## Core API

### Creating a Tokenizer

```go
// Create empty tokenizer
func NewTokenizer() *Tokenizer

// Load tokenizer from GGUF metadata
func NewTokenizerFromGGUF(ggufFile *gguf.GGUFFile) (*Tokenizer, error)
```

**Example**:
```go
// Parse GGUF file
ggufFile, err := gguf.ParseGGUF("model.gguf")
if err != nil {
    panic(err)
}

// Create tokenizer from GGUF metadata
tok, err := tokenizer.NewTokenizerFromGGUF(ggufFile)
if err != nil {
    panic(err)
}

fmt.Printf("Vocabulary size: %d\n", tok.VocabSize())
fmt.Printf("Model type: %s\n", tok.ModelType())
```

### Encoding

```go
// Encode text to token IDs
func (t *Tokenizer) Encode(text string, addBOS, addEOS bool) []int

// Encode text to token strings (for debugging)
func (t *Tokenizer) EncodeAsTokens(text string) []string

// Count tokens in text
func (t *Tokenizer) CountTokens(text string) int
```

**Example**:
```go
// Encode without special tokens
ids := tok.Encode("Hello, world!", false, false)
fmt.Printf("Token IDs: %v\n", ids)

// Encode with BOS and EOS
ids = tok.Encode("Hello, world!", true, true)
// ids = [<BOS>, ...tokens..., <EOS>]

// Debug: see token strings
tokens := tok.EncodeAsTokens("Hello")
fmt.Printf("Tokens: %v\n", tokens)

// Count tokens
count := tok.CountTokens("This is a long text...")
fmt.Printf("Token count: %d\n", count)
```

### Decoding

```go
// Decode token IDs to text
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string

// Decode token strings to text (for debugging)
func (t *Tokenizer) DecodeTokens(tokens []string) string
```

**Example**:
```go
// Decode with special tokens
text := tok.Decode(ids, false)
fmt.Printf("Text: %s\n", text)

// Decode without special tokens
text = tok.Decode(ids, true)
fmt.Printf("Text (no special): %s\n", text)
```

### Vocabulary Access

```go
// Get vocabulary size
func (t *Tokenizer) VocabSize() int

// Convert token string to ID
func (t *Tokenizer) TokenToID(token string) int

// Convert token ID to string
func (t *Tokenizer) IDToToken(id int) string

// Get special token IDs
func (t *Tokenizer) BOSID() int
func (t *Tokenizer) EOSID() int
func (t *Tokenizer) PADID() int
func (t *Tokenizer) UNKID() int

// Get model type
func (t *Tokenizer) ModelType() string
```

### Merge Rules

```go
// Check if a merge rule exists
func (t *Tokenizer) HasMerge(pair string) bool

// Get merge rank (lower = higher priority)
func (t *Tokenizer) GetMergeRank(pair string) int
```

## BPE Algorithm

### Encoding Algorithm

1. **Byte-level preprocessing**: Convert text to UTF-8 bytes
2. **Initial tokenization**: Each byte becomes a token
3. **Iterative merging**: Apply BPE merge rules greedily
4. **Special token handling**: Add BOS/EOS if requested

```
Input: "hello"
Bytes: [h, e, l, l, o]

Initial tokens: ["h", "e", "l", "l", "o"]

Merge iteration 1:
- Find highest-priority merge in current tokens
- Suppose "h e" → "he" (rank 0)
- Result: ["he", "l", "l", "o"]

Merge iteration 2:
- Find "l l" → "ll" (rank 1)
- Result: ["he", "ll", "o"]

Merge iteration 3:
- Find "he ll" → "hell" (rank 2)
- Result: ["hell", "o"]

Merge iteration 4:
- Find "hell o" → "hello" (rank 3)
- Result: ["hello"]

Final: Convert tokens to IDs
```

### Decoding Algorithm

1. **ID to token**: Map each ID to its token string
2. **Concatenation**: Join all tokens
3. **Special token filtering**: Skip BOS/EOS/PAD if requested
4. **UTF-8 reconstruction**: Result is valid UTF-8 string

```
Input: [1, 245, 123, 2]  (where 1=<s>, 2=</s>)

Map to tokens: ["<s>", "hello", "world", "</s>"]

Skip special (if requested): ["hello", "world"]

Concatenate: "helloworld"

Result: "helloworld"
```

## Special Tokens

### Token Types

| Token | Typical ID | Purpose | Example |
|-------|-----------|---------|---------|
| `<unk>` | 0 | Unknown token | Out-of-vocab words |
| `<s>` or `<BOS>` | 1 | Beginning of sequence | Start of prompt |
| `</s>` or `<EOS>` | 2 | End of sequence | End of generation |
| `<pad>` | 3 | Padding | Batch alignment |

### Special Token Handling

**During Encoding**:
- `addBOS=true`: Prepend BOS token
- `addEOS=true`: Append EOS token
- Unknown tokens: Replace with UNK token ID

**During Decoding**:
- `skipSpecial=true`: Skip BOS, EOS, PAD tokens
- `skipSpecial=false`: Include all tokens in output

## Integration with GGUF

### Metadata Keys Used

```go
const (
    KeyTokenizerModel   = "tokenizer.ggml.model"       // "gpt2", "llama"
    KeyTokenizerTokens  = "tokenizer.ggml.tokens"      // Token strings (array)
    KeyTokenizerScores  = "tokenizer.ggml.scores"      // Token scores (array)
    KeyTokenizerMerges  = "tokenizer.ggml.merges"      // BPE merges (array)
    KeyTokenizerBOSID   = "tokenizer.ggml.bos_token_id"
    KeyTokenizerEOSID   = "tokenizer.ggml.eos_token_id"
    KeyTokenizerPADID   = "tokenizer.ggml.padding_token_id"
)
```

### Loading Process

```go
func NewTokenizerFromGGUF(ggufFile *gguf.GGUFFile) (*Tokenizer, error) {
    // 1. Extract token strings
    tokens := ggufFile.GetTokens()

    // 2. Build vocab map (token → ID)
    vocab := make(map[string]int)
    for id, token := range tokens {
        vocab[token] = id
    }

    // 3. Load scores (or default to 0.0)
    scores := ggufFile.GetTokenScores()
    if len(scores) != len(tokens) {
        scores = make([]float32, len(tokens))
    }

    // 4. Load BPE merge rules
    mergeStrs := ggufFile.GetMerges()
    merges := make(map[string]int)
    for rank, merge := range mergeStrs {
        merges[merge] = rank
    }

    // 5. Extract special token IDs
    bosID := extractInt(ggufFile.Metadata[KeyTokenizerBOSID])
    eosID := extractInt(ggufFile.Metadata[KeyTokenizerEOSID])
    // ...

    // 6. Detect UNK token
    unkID := vocab["<unk>"] or vocab["<UNK>"] or -1

    return &Tokenizer{...}
}
```

## Performance Characteristics

### Benchmarks (on typical hardware)

```
BenchmarkEncode:      ~2.3 µs/op   (1376 B/op, 69 allocs)
BenchmarkDecode:      ~78 ns/op    (24 B/op, 2 allocs)
BenchmarkTokenToID:   ~6 ns/op     (0 allocs) - hashmap lookup
BenchmarkIDToToken:   ~0.4 ns/op   (0 allocs) - array index
```

### Complexity Analysis

- **Encode**: O(n * m) where n = text length, m = number of merges
  - Worst case: O(n²) for pathological inputs
  - Typical case: O(n log n) with good merge rules

- **Decode**: O(n) where n = number of tokens
  - Linear time with minimal allocations

- **TokenToID**: O(1) - hashmap lookup
- **IDToToken**: O(1) - array index

### Memory Usage

- **Vocabulary**: O(V) where V = vocab size (~32K-100K tokens)
- **Merges**: O(M) where M = number of merge rules (~50K-100K merges)
- **Per-encode**: O(n) temporary allocations for token list

## Testing

### Test Coverage

- **Coverage**: 100% of statements
- **Tests**: 25+ test cases
- **Categories**:
  - Basic operations (encode, decode, vocab lookup)
  - Special token handling (BOS, EOS, PAD, UNK)
  - Edge cases (empty strings, invalid IDs)
  - GGUF loading (with/without scores, special tokens)
  - UTF-8 handling (ASCII, Unicode, emojis)

### Example Test

```go
func TestEncode_WithSpecialTokens(t *testing.T) {
    tok := createMockTokenizer()

    ids := tok.Encode("hello", true, true)

    // First token should be BOS
    if ids[0] != tok.BOSID() {
        t.Errorf("Expected BOS token at start")
    }

    // Last token should be EOS
    if ids[len(ids)-1] != tok.EOSID() {
        t.Errorf("Expected EOS token at end")
    }
}
```

## Special Token Encoding (Chat Templates)

### Problem

Special tokens like `<|im_start|>`, `<|im_end|>`, `<|eot_id|>` used in chat templates
must encode as single token IDs. The standard BPE path splits them into individual
characters (`<`, `|`, `i`, `m`, ...) which may not merge back correctly.

### Solution: Pre-tokenization Split

When the input text contains `<|` (a cheap check), the encoder uses
`encodeWithSpecialTokenSplit()` which:

1. **Lazily builds** a sorted list of special tokens from the vocab (tokens matching `<|...|>`)
2. **Scans** the text for occurrences of known special tokens
3. **Splits** text into alternating segments: `[regular_text, special_token, regular_text, ...]`
4. **Encodes** regular segments through the standard BPE path
5. **Looks up** special token IDs directly via `t.vocab[]`

```go
// Cached lazily on first use
type Tokenizer struct {
    // ... existing fields ...
    specialTokens []string // sorted by length descending
}

// Called automatically when text contains "<|"
func (t *Tokenizer) encodeWithSpecialTokenSplit(text string, addBOS, addEOS bool) []int
```

### Example

```
Input: "<|im_start|>user\nhello<|im_end|>\n"

Split segments:
  1. ""              → (empty, skip)
  2. "<|im_start|>"  → lookup: token ID 151644
  3. "user\nhello"   → BPE encode: [882, 198, 15339]
  4. "<|im_end|>"    → lookup: token ID 151645
  5. "\n"            → BPE encode: [198]

Result: [151644, 882, 198, 15339, 151645, 198]
```

## Limitations

1. **Byte-level BPE only**: Does not support character-level or SentencePiece
2. **Greedy merging**: Always applies highest-priority merge first
3. **No normalization**: Text preprocessing (lowercasing, etc.) not included
4. **No regex splitting**: Does not handle pre-tokenization patterns

## Future Enhancements

1. **SentencePiece support**: Add unigram tokenization
2. **Pre-tokenization**: Support regex patterns for splitting
3. **Normalization**: Add text preprocessing options
4. **Streaming encoding**: Process text incrementally
5. **Parallel encoding**: Multi-threaded batch encoding

## References

- [BPE Paper](https://arxiv.org/abs/1508.07909) - Neural Machine Translation of Rare Words with Subword Units
- [llama.cpp Tokenizer](https://github.com/ggml-org/llama.cpp) - Reference implementation
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizer library
- [GGUF Format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) - Model file format specification
