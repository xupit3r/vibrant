# GGUF Format Specification

## Overview

GGUF (GPT-Generated Unified Format) is a binary format for storing LLM models with quantization support. The `internal/gguf/` package provides parsing and loading capabilities for GGUF files used by llama.cpp and compatible tools.

## Design Goals

1. **Lazy Loading**: Only load tensors when needed to minimize memory usage
2. **Memory Mapping**: Use mmap for efficient access to large files
3. **Metadata Access**: Extract model configuration without loading weights
4. **Validation**: Verify file integrity and version compatibility

## File Format Structure

### GGUF Binary Layout

```
┌────────────────────────────────────────┐
│         Header (12-16 bytes)           │
│  - Magic: "GGUF" (0x46554747)          │
│  - Version: uint32                     │
│  - Tensor count: uint64                │
│  - Metadata KV count: uint64           │
├────────────────────────────────────────┤
│          Metadata Key-Value            │
│  (Repeated metadata_kv_count times)    │
│  - Key: string                         │
│  - Value type: uint32                  │
│  - Value: (type-dependent)             │
├────────────────────────────────────────┤
│         Tensor Information             │
│  (Repeated tensor_count times)         │
│  - Name: string                        │
│  - Dimensions: uint32[]                │
│  - Type: uint32 (ggml_type)            │
│  - Offset: uint64                      │
├────────────────────────────────────────┤
│          Padding/Alignment             │
│  (Align to 32-byte boundary)           │
├────────────────────────────────────────┤
│           Tensor Data                  │
│  (Raw tensor bytes at offsets)         │
└────────────────────────────────────────┘
```

### Version History

- **Version 1**: Initial format (deprecated)
- **Version 2**: Added tensor offset alignment
- **Version 3**: Current stable version (llama.cpp default)

## Data Structures

### GGUFFile

```go
type GGUFFile struct {
    path        string
    file        *os.File
    magic       uint32
    version     uint32
    tensorCount uint64
    kvCount     uint64

    Metadata    map[string]interface{}
    Tensors     map[string]*TensorInfo

    // Offset where tensor data begins
    tensorDataOffset int64
}

// TensorInfo describes a tensor in the file
type TensorInfo struct {
    Name   string
    Dims   []uint64
    Type   GGMLType
    Offset uint64        // Offset from tensorDataOffset
    Size   uint64        // Size in bytes
}

// GGMLType represents quantization types
type GGMLType uint32
const (
    GGML_TYPE_F32  GGMLType = 0   // float32
    GGML_TYPE_F16  GGMLType = 1   // float16
    GGML_TYPE_Q4_0 GGMLType = 2   // 4-bit quantization (type 0)
    GGML_TYPE_Q4_1 GGMLType = 3   // 4-bit quantization (type 1)
    GGML_TYPE_Q5_0 GGMLType = 6   // 5-bit quantization (type 0)
    GGML_TYPE_Q5_1 GGMLType = 7   // 5-bit quantization (type 1)
    GGML_TYPE_Q8_0 GGMLType = 8   // 8-bit quantization
    GGML_TYPE_Q4_K GGMLType = 12  // 4-bit k-quant
    GGML_TYPE_Q5_K GGMLType = 13  // 5-bit k-quant
    GGML_TYPE_Q6_K GGMLType = 14  // 6-bit k-quant
    GGML_TYPE_Q8_K GGMLType = 15  // 8-bit k-quant
)

// String returns human-readable type name (added in Phase 10.10)
func (t GGMLType) String() string {
    switch t {
    case GGML_TYPE_F32:
        return "F32"
    case GGML_TYPE_F16:
        return "F16"
    case GGML_TYPE_Q4_K:
        return "Q4_K"
    case GGML_TYPE_Q5_K:
        return "Q5_K"
    case GGML_TYPE_Q6_K:
        return "Q6_K"
    // ... other types
    default:
        return fmt.Sprintf("Unknown(%d)", t)
    }
}
```

### Metadata Value Types

```go
type ValueType uint32
const (
    GGUF_METADATA_VALUE_TYPE_UINT8   ValueType = 0
    GGUF_METADATA_VALUE_TYPE_INT8    ValueType = 1
    GGUF_METADATA_VALUE_TYPE_UINT16  ValueType = 2
    GGUF_METADATA_VALUE_TYPE_INT16   ValueType = 3
    GGUF_METADATA_VALUE_TYPE_UINT32  ValueType = 4
    GGUF_METADATA_VALUE_TYPE_INT32   ValueType = 5
    GGUF_METADATA_VALUE_TYPE_FLOAT32 ValueType = 6
    GGUF_METADATA_VALUE_TYPE_BOOL    ValueType = 7
    GGUF_METADATA_VALUE_TYPE_STRING  ValueType = 8
    GGUF_METADATA_VALUE_TYPE_ARRAY   ValueType = 9
    GGUF_METADATA_VALUE_TYPE_UINT64  ValueType = 10
    GGUF_METADATA_VALUE_TYPE_INT64   ValueType = 11
    GGUF_METADATA_VALUE_TYPE_FLOAT64 ValueType = 12
)
```

## Parser Implementation

### Opening a GGUF File

```go
func ParseGGUF(path string) (*GGUFFile, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, fmt.Errorf("failed to open GGUF file: %w", err)
    }

    gguf := &GGUFFile{
        path:     path,
        file:     f,
        Metadata: make(map[string]interface{}),
        Tensors:  make(map[string]*TensorInfo),
    }

    // Parse header
    if err := gguf.parseHeader(); err != nil {
        f.Close()
        return nil, err
    }

    // Parse metadata
    if err := gguf.parseMetadata(); err != nil {
        f.Close()
        return nil, err
    }

    // Parse tensor info
    if err := gguf.parseTensorInfo(); err != nil {
        f.Close()
        return nil, err
    }

    return gguf, nil
}
```

### Parsing Header

```go
func (g *GGUFFile) parseHeader() error {
    r := bufio.NewReader(g.file)

    // Read magic (4 bytes)
    if err := binary.Read(r, binary.LittleEndian, &g.magic); err != nil {
        return err
    }
    if g.magic != 0x46554747 { // "GGUF"
        return fmt.Errorf("invalid magic: expected GGUF, got 0x%x", g.magic)
    }

    // Read version (4 bytes)
    if err := binary.Read(r, binary.LittleEndian, &g.version); err != nil {
        return err
    }
    if g.version < 2 || g.version > 3 {
        return fmt.Errorf("unsupported GGUF version: %d", g.version)
    }

    // Read counts (16 bytes)
    if err := binary.Read(r, binary.LittleEndian, &g.tensorCount); err != nil {
        return err
    }
    if err := binary.Read(r, binary.LittleEndian, &g.kvCount); err != nil {
        return err
    }

    return nil
}
```

### Parsing Metadata

```go
func (g *GGUFFile) parseMetadata() error {
    r := bufio.NewReader(g.file)

    for i := uint64(0); i < g.kvCount; i++ {
        // Read key (length-prefixed string)
        key, err := g.readString(r)
        if err != nil {
            return fmt.Errorf("failed to read metadata key %d: %w", i, err)
        }

        // Read value type
        var valueType ValueType
        if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
            return err
        }

        // Read value based on type
        value, err := g.readMetadataValue(r, valueType)
        if err != nil {
            return fmt.Errorf("failed to read metadata value for %s: %w", key, err)
        }

        g.Metadata[key] = value
    }

    return nil
}

func (g *GGUFFile) readMetadataValue(r io.Reader, vtype ValueType) (interface{}, error) {
    switch vtype {
    case GGUF_METADATA_VALUE_TYPE_UINT32:
        var v uint32
        err := binary.Read(r, binary.LittleEndian, &v)
        return v, err

    case GGUF_METADATA_VALUE_TYPE_INT32:
        var v int32
        err := binary.Read(r, binary.LittleEndian, &v)
        return v, err

    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
        var v float32
        err := binary.Read(r, binary.LittleEndian, &v)
        return v, err

    case GGUF_METADATA_VALUE_TYPE_STRING:
        return g.readString(r)

    case GGUF_METADATA_VALUE_TYPE_ARRAY:
        return g.readArray(r)

    case GGUF_METADATA_VALUE_TYPE_BOOL:
        var v uint8
        err := binary.Read(r, binary.LittleEndian, &v)
        return v != 0, err

    default:
        return nil, fmt.Errorf("unsupported metadata type: %d", vtype)
    }
}

func (g *GGUFFile) readString(r io.Reader) (string, error) {
    // Read length (uint64)
    var length uint64
    if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
        return "", err
    }

    // Read string bytes
    buf := make([]byte, length)
    if _, err := io.ReadFull(r, buf); err != nil {
        return "", err
    }

    return string(buf), nil
}

func (g *GGUFFile) readArray(r io.Reader) ([]interface{}, error) {
    // Read element type
    var elemType ValueType
    if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
        return nil, err
    }

    // Read array length
    var length uint64
    if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
        return nil, err
    }

    // Read elements
    arr := make([]interface{}, length)
    for i := uint64(0); i < length; i++ {
        val, err := g.readMetadataValue(r, elemType)
        if err != nil {
            return nil, err
        }
        arr[i] = val
    }

    return arr, nil
}
```

### Parsing Tensor Information

```go
func (g *GGUFFile) parseTensorInfo() error {
    r := bufio.NewReader(g.file)

    for i := uint64(0); i < g.tensorCount; i++ {
        info := &TensorInfo{}

        // Read tensor name
        name, err := g.readString(r)
        if err != nil {
            return err
        }
        info.Name = name

        // Read number of dimensions
        var nDims uint32
        if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
            return err
        }

        // Read dimensions
        info.Dims = make([]uint64, nDims)
        for j := uint32(0); j < nDims; j++ {
            if err := binary.Read(r, binary.LittleEndian, &info.Dims[j]); err != nil {
                return err
            }
        }

        // Read tensor type
        if err := binary.Read(r, binary.LittleEndian, &info.Type); err != nil {
            return err
        }

        // Read offset
        if err := binary.Read(r, binary.LittleEndian, &info.Offset); err != nil {
            return err
        }

        // Calculate size
        info.Size = g.calculateTensorSize(info.Dims, info.Type)

        g.Tensors[name] = info
    }

    // Record offset where tensor data begins
    offset, _ := g.file.Seek(0, io.SeekCurrent)
    // Align to 32-byte boundary
    g.tensorDataOffset = (offset + 31) &^ 31

    return nil
}

func (g *GGUFFile) calculateTensorSize(dims []uint64, dtype GGMLType) uint64 {
    // Calculate total number of elements
    elements := uint64(1)
    for _, dim := range dims {
        elements *= dim
    }

    // Calculate size based on type
    switch dtype {
    case GGML_TYPE_F32:
        return elements * 4
    case GGML_TYPE_F16:
        return elements * 2
    case GGML_TYPE_Q4_0:
        // 32 values per block, each block is 18 bytes
        return ((elements + 31) / 32) * 18
    case GGML_TYPE_Q4_K:
        // 256 values per block, each block is 144 bytes
        return ((elements + 255) / 256) * 144
    case GGML_TYPE_Q5_K:
        // 256 values per block, each block is 176 bytes
        return ((elements + 255) / 256) * 176
    case GGML_TYPE_Q8_0:
        // 32 values per block, each block is 34 bytes
        return ((elements + 31) / 32) * 34
    default:
        return 0
    }
}
```

## Tensor Loading

### Lazy Loading with Memory Mapping

```go
func (g *GGUFFile) LoadTensor(name string) (*tensor.Tensor, error) {
    info, ok := g.Tensors[name]
    if !ok {
        return nil, fmt.Errorf("tensor %s not found", name)
    }

    // Calculate absolute offset in file
    absoluteOffset := g.tensorDataOffset + int64(info.Offset)

    // Convert GGML type to tensor DataType
    dtype := g.ggmlTypeToTensorType(info.Type)

    // Convert dimensions to int slice
    shape := make([]int, len(info.Dims))
    for i, dim := range info.Dims {
        shape[i] = int(dim)
    }

    // Memory-map the tensor data
    t, err := tensor.NewTensorMmap(g.path, absoluteOffset, int64(info.Size), shape, dtype)
    if err != nil {
        return nil, fmt.Errorf("failed to mmap tensor %s: %w", name, err)
    }

    return t, nil
}

func (g *GGUFFile) ggmlTypeToTensorType(gtype GGMLType) tensor.DataType {
    switch gtype {
    case GGML_TYPE_F32:
        return tensor.Float32
    case GGML_TYPE_F16:
        return tensor.Float16
    case GGML_TYPE_Q4_K:
        return tensor.Q4_K
    case GGML_TYPE_Q5_K:
        return tensor.Q5_K
    case GGML_TYPE_Q8_0:
        return tensor.Q8_0
    default:
        return tensor.Float32
    }
}
```

### Eager Loading (Optional)

```go
func (g *GGUFFile) LoadTensorEager(name string) (*tensor.Tensor, error) {
    info, ok := g.Tensors[name]
    if !ok {
        return nil, fmt.Errorf("tensor %s not found", name)
    }

    // Seek to tensor data
    absoluteOffset := g.tensorDataOffset + int64(info.Offset)
    if _, err := g.file.Seek(absoluteOffset, io.SeekStart); err != nil {
        return nil, err
    }

    // Read data into memory
    data := make([]byte, info.Size)
    if _, err := io.ReadFull(g.file, data); err != nil {
        return nil, err
    }

    // Convert to tensor based on type
    shape := make([]int, len(info.Dims))
    for i, dim := range info.Dims {
        shape[i] = int(dim)
    }

    dtype := g.ggmlTypeToTensorType(info.Type)

    // Wrap data in tensor
    return tensor.NewTensorFromBytes(data, shape, dtype)
}
```

## Metadata Extraction

### Common Metadata Keys

```go
// Architecture and model type
const (
    KeyArchitecture     = "general.architecture"      // "llama", "qwen", "mistral"
    KeyName             = "general.name"               // Model name
    KeyFileType         = "general.file_type"          // Quantization type

    // Model hyperparameters (llama/qwen)
    KeyContextLength    = "%s.context_length"          // Max sequence length
    KeyEmbeddingLength  = "%s.embedding_length"        // Hidden dimension
    KeyBlockCount       = "%s.block_count"             // Number of layers
    KeyAttentionHeadCount = "%s.attention.head_count"  // Number of attention heads
    KeyAttentionHeadCountKV = "%s.attention.head_count_kv" // KV heads (GQA)
    KeyFFNLength        = "%s.feed_forward_length"     // FFN intermediate size
    KeyRopeFreqBase     = "%s.rope.freq_base"          // RoPE frequency base
    KeyNormRMSEps       = "%s.attention.layer_norm_rms_epsilon" // RMSNorm epsilon

    // Tokenizer
    KeyTokenizerModel   = "tokenizer.ggml.model"       // "gpt2", "llama"
    KeyTokenizerTokens  = "tokenizer.ggml.tokens"      // Token strings (array)
    KeyTokenizerScores  = "tokenizer.ggml.scores"      // Token scores (array)
    KeyTokenizerMerges  = "tokenizer.ggml.merges"      // BPE merges (array)
    KeyTokenizerBOSID   = "tokenizer.ggml.bos_token_id"
    KeyTokenizerEOSID   = "tokenizer.ggml.eos_token_id"
    KeyTokenizerPADID   = "tokenizer.ggml.padding_token_id"
)
```

### Helper Functions

```go
// GetArchitecture returns the model architecture (llama, qwen, etc.)
func (g *GGUFFile) GetArchitecture() string {
    if arch, ok := g.Metadata[KeyArchitecture].(string); ok {
        return arch
    }
    return "unknown"
}

// GetMetadataString gets a string metadata value with architecture substitution
func (g *GGUFFile) GetMetadataString(key string) (string, bool) {
    arch := g.GetArchitecture()
    fullKey := fmt.Sprintf(key, arch)
    val, ok := g.Metadata[fullKey].(string)
    return val, ok
}

// GetMetadataInt gets an integer metadata value
func (g *GGUFFile) GetMetadataInt(key string) (int, bool) {
    arch := g.GetArchitecture()
    fullKey := fmt.Sprintf(key, arch)

    switch v := g.Metadata[fullKey].(type) {
    case int32:
        return int(v), true
    case uint32:
        return int(v), true
    case int64:
        return int(v), true
    case uint64:
        return int(v), true
    default:
        return 0, false
    }
}

// GetTokens extracts tokenizer vocabulary
func (g *GGUFFile) GetTokens() []string {
    if tokens, ok := g.Metadata[KeyTokenizerTokens].([]interface{}); ok {
        result := make([]string, len(tokens))
        for i, t := range tokens {
            if s, ok := t.(string); ok {
                result[i] = s
            }
        }
        return result
    }
    return nil
}
```

## Usage Example

```go
package main

import (
    "fmt"
    "vibrant/internal/gguf"
)

func main() {
    // Open GGUF file
    file, err := gguf.ParseGGUF("/path/to/qwen-2.5-coder-3b-q4_k_m.gguf")
    if err != nil {
        panic(err)
    }
    defer file.Close()

    // Extract metadata
    arch := file.GetArchitecture()
    fmt.Printf("Architecture: %s\n", arch)

    contextLen, _ := file.GetMetadataInt(gguf.KeyContextLength)
    fmt.Printf("Context length: %d\n", contextLen)

    hiddenDim, _ := file.GetMetadataInt(gguf.KeyEmbeddingLength)
    fmt.Printf("Hidden dimension: %d\n", hiddenDim)

    numLayers, _ := file.GetMetadataInt(gguf.KeyBlockCount)
    fmt.Printf("Layers: %d\n", numLayers)

    // List all tensors
    fmt.Printf("\nTensors (%d total):\n", len(file.Tensors))
    for name, info := range file.Tensors {
        fmt.Printf("  %s: %v (type=%d, size=%d bytes)\n",
            name, info.Dims, info.Type, info.Size)
    }

    // Load a specific tensor (lazy, memory-mapped)
    embedTensor, err := file.LoadTensor("token_embd.weight")
    if err != nil {
        panic(err)
    }
    fmt.Printf("\nLoaded embedding tensor: shape=%v\n", embedTensor.Shape())

    // Get tokenizer vocabulary
    tokens := file.GetTokens()
    fmt.Printf("\nVocabulary size: %d\n", len(tokens))
    fmt.Printf("First 10 tokens: %v\n", tokens[:10])
}
```

## Testing

### Test Data

```go
func TestParseGGUF(t *testing.T) {
    // Use a small test GGUF file (or mock one)
    file, err := gguf.ParseGGUF("testdata/tiny-model-q4_k.gguf")
    if err != nil {
        t.Fatalf("Failed to parse GGUF: %v", err)
    }
    defer file.Close()

    // Verify header
    if file.Version() < 2 {
        t.Errorf("Expected version >= 2, got %d", file.Version())
    }

    // Verify metadata
    arch := file.GetArchitecture()
    if arch == "unknown" {
        t.Error("Failed to extract architecture")
    }

    // Verify tensors
    if len(file.Tensors) == 0 {
        t.Error("No tensors found")
    }
}
```

## Performance Considerations

1. **Memory Mapping**: Use mmap for all tensor loads to avoid copying large amounts of data
2. **Lazy Loading**: Only load tensors as needed, not all at once
3. **Metadata Caching**: Cache frequently accessed metadata values
4. **Alignment**: Respect 32-byte alignment for SIMD operations

## References

- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp GGUF Implementation](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf.py)
- [HuggingFace GGUF Documentation](https://huggingface.co/docs/hub/gguf)
