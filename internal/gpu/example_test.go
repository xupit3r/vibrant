package gpu_test

import (
	"fmt"
	"log"

	"github.com/xupit3r/vibrant/internal/gpu"
)

// Example of using the GPU device abstraction
func Example_basicUsage() {
	// Get default device (GPU on macOS, CPU otherwise)
	dev, err := gpu.GetDefaultDevice()
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Free()

	fmt.Printf("Device: %s (%s)\n", dev.Name(), dev.Type())

	// Allocate a buffer on the device
	size := int64(1024)
	buf, err := dev.Allocate(size)
	if err != nil {
		log.Fatal(err)
	}
	defer buf.Free()

	// Create some test data
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}

	// Copy data to device
	if err := buf.CopyFromHost(data); err != nil {
		log.Fatal(err)
	}

	// Copy data back from device
	result := make([]byte, size)
	if err := buf.CopyToHost(result); err != nil {
		log.Fatal(err)
	}

	// Wait for all operations to complete
	if err := dev.Sync(); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Data transfer successful")
}

// Example of explicitly selecting a GPU device
func Example_gpuDevice() {
	dev, err := gpu.GetDevice(gpu.DeviceTypeGPU)
	if err != nil {
		log.Printf("GPU not available: %v", err)
		return
	}
	defer dev.Free()

	// Check memory usage
	used, total := dev.MemoryUsage()
	fmt.Printf("GPU Memory: %d MB / %d MB\n", used/(1024*1024), total/(1024*1024))
}

// Example of copying between buffers on the same device
func Example_bufferCopy() {
	dev, err := gpu.GetDefaultDevice()
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Free()

	size := int64(4096)

	// Allocate two buffers
	src, err := dev.Allocate(size)
	if err != nil {
		log.Fatal(err)
	}
	defer src.Free()

	dst, err := dev.Allocate(size)
	if err != nil {
		log.Fatal(err)
	}
	defer dst.Free()

	// Fill source buffer
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i)
	}
	src.CopyFromHost(data)

	// Copy from src to dst on the device
	if err := dev.Copy(dst, src, size); err != nil {
		log.Fatal(err)
	}

	// Sync to ensure copy is complete
	dev.Sync()

	// Verify the copy
	result := make([]byte, size)
	dst.CopyToHost(result)

	fmt.Println("Buffer copy successful")
}
