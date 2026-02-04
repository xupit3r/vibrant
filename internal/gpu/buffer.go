package gpu

// Buffer represents a GPU memory buffer
type Buffer interface {
	// Size returns the size of the buffer in bytes
	Size() int64

	// Ptr returns the raw pointer to the buffer (for GPU APIs)
	// Returns 0 for CPU buffers
	Ptr() uintptr

	// CopyToHost copies buffer data to host memory
	CopyToHost(dst []byte) error

	// CopyFromHost copies host memory to the buffer
	CopyFromHost(src []byte) error

	// Free releases the buffer
	Free() error

	// Device returns the device that owns this buffer
	Device() Device
}
