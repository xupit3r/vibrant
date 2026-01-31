package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// GGUF magic number: "GGUF" in little-endian
const ggufMagic = 0x46554747

// ParseGGUF opens and parses a GGUF file
func ParseGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open GGUF file: %w", err)
	}

	gguf := &GGUFFile{
		path:     path,
		Metadata: make(map[string]interface{}),
		Tensors:  make(map[string]*TensorInfo),
	}

	// Create buffered reader for efficient reading
	r := bufio.NewReader(f)

	// Parse header
	if err := gguf.parseHeader(r); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Parse metadata
	if err := gguf.parseMetadata(r); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Parse tensor information
	if err := gguf.parseTensorInfo(r); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to parse tensor info: %w", err)
	}

	// Calculate tensor data offset (aligned to 32 bytes)
	// Get current position
	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to get file position: %w", err)
	}

	// Account for buffered data
	offset -= int64(r.Buffered())

	// Align to 32-byte boundary
	gguf.tensorDataOffset = (offset + 31) &^ 31

	f.Close()
	return gguf, nil
}

// parseHeader reads the GGUF file header
func (g *GGUFFile) parseHeader(r io.Reader) error {
	// Read magic (4 bytes)
	if err := binary.Read(r, binary.LittleEndian, &g.magic); err != nil {
		return fmt.Errorf("failed to read magic: %w", err)
	}
	if g.magic != ggufMagic {
		return fmt.Errorf("invalid magic: expected 0x%x (GGUF), got 0x%x", ggufMagic, g.magic)
	}

	// Read version (4 bytes)
	if err := binary.Read(r, binary.LittleEndian, &g.version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}
	if g.version < 2 || g.version > 3 {
		return fmt.Errorf("unsupported GGUF version: %d (supported: 2-3)", g.version)
	}

	// Read tensor count (8 bytes)
	if err := binary.Read(r, binary.LittleEndian, &g.tensorCount); err != nil {
		return fmt.Errorf("failed to read tensor count: %w", err)
	}

	// Read metadata KV count (8 bytes)
	if err := binary.Read(r, binary.LittleEndian, &g.kvCount); err != nil {
		return fmt.Errorf("failed to read KV count: %w", err)
	}

	return nil
}

// parseMetadata reads all metadata key-value pairs
func (g *GGUFFile) parseMetadata(r io.Reader) error {
	for i := uint64(0); i < g.kvCount; i++ {
		// Read key (length-prefixed string)
		key, err := readString(r)
		if err != nil {
			return fmt.Errorf("failed to read metadata key %d: %w", i, err)
		}

		// Read value type
		var valueType ValueType
		if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
			return fmt.Errorf("failed to read value type for key %s: %w", key, err)
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

// readMetadataValue reads a metadata value based on its type
func (g *GGUFFile) readMetadataValue(r io.Reader, vtype ValueType) (interface{}, error) {
	switch vtype {
	case GGUF_METADATA_VALUE_TYPE_UINT8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_INT8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_UINT16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_INT16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

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

	case GGUF_METADATA_VALUE_TYPE_UINT64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_INT64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_FLOAT64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err

	case GGUF_METADATA_VALUE_TYPE_BOOL:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err

	case GGUF_METADATA_VALUE_TYPE_STRING:
		return readString(r)

	case GGUF_METADATA_VALUE_TYPE_ARRAY:
		return g.readArray(r)

	default:
		return nil, fmt.Errorf("unsupported metadata type: %d", vtype)
	}
}

// readString reads a length-prefixed string
func readString(r io.Reader) (string, error) {
	// Read length (uint64)
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("failed to read string length: %w", err)
	}

	// Sanity check: prevent excessive allocations
	if length > 1<<30 { // 1GB limit
		return "", fmt.Errorf("string length too large: %d", length)
	}

	// Read string bytes
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("failed to read string data: %w", err)
	}

	return string(buf), nil
}

// readArray reads an array of values
func (g *GGUFFile) readArray(r io.Reader) ([]interface{}, error) {
	// Read element type
	var elemType ValueType
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, fmt.Errorf("failed to read array element type: %w", err)
	}

	// Read array length
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, fmt.Errorf("failed to read array length: %w", err)
	}

	// Sanity check
	if length > 1<<28 { // 256M elements limit
		return nil, fmt.Errorf("array length too large: %d", length)
	}

	// Read elements
	arr := make([]interface{}, length)
	for i := uint64(0); i < length; i++ {
		val, err := g.readMetadataValue(r, elemType)
		if err != nil {
			return nil, fmt.Errorf("failed to read array element %d: %w", i, err)
		}
		arr[i] = val
	}

	return arr, nil
}

// parseTensorInfo reads tensor information for all tensors
func (g *GGUFFile) parseTensorInfo(r io.Reader) error {
	for i := uint64(0); i < g.tensorCount; i++ {
		info := &TensorInfo{}

		// Read tensor name
		name, err := readString(r)
		if err != nil {
			return fmt.Errorf("failed to read tensor name %d: %w", i, err)
		}
		info.Name = name

		// Read number of dimensions
		var nDims uint32
		if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
			return fmt.Errorf("failed to read dimension count for %s: %w", name, err)
		}

		// Sanity check
		if nDims > 16 {
			return fmt.Errorf("too many dimensions for tensor %s: %d", name, nDims)
		}

		// Read dimensions
		info.Dims = make([]uint64, nDims)
		for j := uint32(0); j < nDims; j++ {
			if err := binary.Read(r, binary.LittleEndian, &info.Dims[j]); err != nil {
				return fmt.Errorf("failed to read dimension %d for %s: %w", j, name, err)
			}
		}

		// Read tensor type
		if err := binary.Read(r, binary.LittleEndian, &info.Type); err != nil {
			return fmt.Errorf("failed to read type for %s: %w", name, err)
		}

		// Read offset
		if err := binary.Read(r, binary.LittleEndian, &info.Offset); err != nil {
			return fmt.Errorf("failed to read offset for %s: %w", name, err)
		}

		// Calculate size
		info.Size = calculateTensorSize(info.Dims, info.Type)

		g.Tensors[name] = info
	}

	return nil
}
