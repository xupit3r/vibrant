package gguf

import (
	"fmt"
	
	"github.com/xupit3r/vibrant/internal/tensor"
)

// GGMLType represents tensor data types in GGUF files
type GGMLType uint32

// GGML tensor type constants
const (
	GGML_TYPE_F32  GGMLType = 0  // float32
	GGML_TYPE_F16  GGMLType = 1  // float16
	GGML_TYPE_Q4_0 GGMLType = 2  // 4-bit quantization (type 0)
	GGML_TYPE_Q4_1 GGMLType = 3  // 4-bit quantization (type 1)
	GGML_TYPE_Q5_0 GGMLType = 6  // 5-bit quantization (type 0)
	GGML_TYPE_Q5_1 GGMLType = 7  // 5-bit quantization (type 1)
	GGML_TYPE_Q8_0 GGMLType = 8  // 8-bit quantization
	GGML_TYPE_Q4_K GGMLType = 12 // 4-bit k-quant
	GGML_TYPE_Q5_K GGMLType = 13 // 5-bit k-quant
	GGML_TYPE_Q6_K GGMLType = 14 // 6-bit k-quant
	GGML_TYPE_Q8_K GGMLType = 15 // 8-bit k-quant
)

// String returns the string representation of the GGML type
func (g GGMLType) String() string {
	switch g {
	case GGML_TYPE_F32:
		return "F32"
	case GGML_TYPE_F16:
		return "F16"
	case GGML_TYPE_Q4_0:
		return "Q4_0"
	case GGML_TYPE_Q4_1:
		return "Q4_1"
	case GGML_TYPE_Q5_0:
		return "Q5_0"
	case GGML_TYPE_Q5_1:
		return "Q5_1"
	case GGML_TYPE_Q8_0:
		return "Q8_0"
	case GGML_TYPE_Q4_K:
		return "Q4_K"
	case GGML_TYPE_Q5_K:
		return "Q5_K"
	case GGML_TYPE_Q6_K:
		return "Q6_K"
	case GGML_TYPE_Q8_K:
		return "Q8_K"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", g)
	}
}

// ValueType represents metadata value types in GGUF files
type ValueType uint32

// GGUF metadata value type constants
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

// Common metadata keys for GGUF files
const (
	// Architecture and model type
	KeyArchitecture          = "general.architecture"                     // "llama", "qwen", "mistral"
	KeyName                  = "general.name"                              // Model name
	KeyFileType              = "general.file_type"                         // Quantization type
	KeyContextLength         = "%s.context_length"                         // Max sequence length
	KeyEmbeddingLength       = "%s.embedding_length"                       // Hidden dimension
	KeyBlockCount            = "%s.block_count"                            // Number of layers
	KeyAttentionHeadCount    = "%s.attention.head_count"                   // Number of attention heads
	KeyAttentionHeadCountKV  = "%s.attention.head_count_kv"                // KV heads (GQA)
	KeyFFNLength             = "%s.feed_forward_length"                    // FFN intermediate size
	KeyRopeFreqBase          = "%s.rope.freq_base"                         // RoPE frequency base
	KeyNormRMSEps            = "%s.attention.layer_norm_rms_epsilon"       // RMSNorm epsilon
	KeyTokenizerModel        = "tokenizer.ggml.model"                      // "gpt2", "llama"
	KeyTokenizerTokens       = "tokenizer.ggml.tokens"                     // Token strings (array)
	KeyTokenizerScores       = "tokenizer.ggml.scores"                     // Token scores (array)
	KeyTokenizerMerges       = "tokenizer.ggml.merges"                     // BPE merges (array)
	KeyTokenizerBOSID        = "tokenizer.ggml.bos_token_id"               // BOS token ID
	KeyTokenizerEOSID        = "tokenizer.ggml.eos_token_id"               // EOS token ID
	KeyTokenizerPADID        = "tokenizer.ggml.padding_token_id"           // Padding token ID
	KeyChatTemplate          = "tokenizer.chat_template"                   // Chat template (Jinja2)
)

// TensorInfo describes a tensor in the GGUF file
type TensorInfo struct {
	Name   string   // Tensor name (e.g., "token_embd.weight")
	Dims   []uint64 // Tensor dimensions
	Type   GGMLType // Tensor data type
	Offset uint64   // Offset from tensorDataOffset
	Size   uint64   // Size in bytes
}

// GGUFFile represents a parsed GGUF file
type GGUFFile struct {
	path    string
	magic   uint32
	version uint32

	tensorCount uint64
	kvCount     uint64

	// Metadata key-value pairs
	Metadata map[string]interface{}

	// Tensor information (name -> info)
	Tensors map[string]*TensorInfo

	// Offset where tensor data begins (after alignment)
	tensorDataOffset int64
}

// Version returns the GGUF file version
func (g *GGUFFile) Version() uint32 {
	return g.version
}

// Path returns the file path
func (g *GGUFFile) Path() string {
	return g.path
}

// TensorCount returns the number of tensors in the file
func (g *GGUFFile) TensorCount() uint64 {
	return g.tensorCount
}

// MetadataCount returns the number of metadata key-value pairs
func (g *GGUFFile) MetadataCount() uint64 {
	return g.kvCount
}

// ggmlTypeToTensorType converts GGML type to tensor DataType
func ggmlTypeToTensorType(gtype GGMLType) tensor.DataType {
	switch gtype {
	case GGML_TYPE_F32:
		return tensor.Float32
	case GGML_TYPE_F16:
		return tensor.Float16
	case GGML_TYPE_Q4_K:
		return tensor.Q4_K
	case GGML_TYPE_Q5_K:
		return tensor.Q5_K
	case GGML_TYPE_Q6_K:
		return tensor.Q6_K
	case GGML_TYPE_Q8_0:
		return tensor.Q8_0
	default:
		return tensor.Float32
	}
}

// calculateTensorSize calculates the size in bytes for a tensor
func calculateTensorSize(dims []uint64, dtype GGMLType) uint64 {
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
	case GGML_TYPE_Q6_K:
		// 256 values per block, each block is 210 bytes
		return ((elements + 255) / 256) * 210
	case GGML_TYPE_Q8_0:
		// 32 values per block, each block is 34 bytes
		return ((elements + 31) / 32) * 34
	case GGML_TYPE_Q8_K:
		// 256 values per block, each block is 292 bytes
		return ((elements + 255) / 256) * 292
	default:
		return 0
	}
}
