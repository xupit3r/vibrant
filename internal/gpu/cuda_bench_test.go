// +build linux,cgo

package gpu

import (
	"fmt"
	"testing"
)

// BenchmarkCUDAAllocation benchmarks buffer allocation with pool
func BenchmarkCUDAAllocation(b *testing.B) {
	dev, err := NewCUDADevice()
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	size := int64(1024 * 1024) // 1 MB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, err := dev.Allocate(size)
		if err != nil {
			b.Fatalf("Allocation failed: %v", err)
		}
		buf.Free()
	}
	b.StopTimer()

	stats := dev.PoolStats()
	b.ReportMetric(float64(stats.PoolHits), "pool-hits")
	b.ReportMetric(float64(stats.PoolMisses), "pool-misses")
	b.ReportMetric(float64(stats.Evictions), "pool-evictions")
}

// BenchmarkCUDADirectAllocation benchmarks direct allocation without pool
func BenchmarkCUDADirectAllocation(b *testing.B) {
	dev, err := NewCUDADevice()
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	// Disable pool by setting it to nil
	dev.pool = nil

	size := int64(1024 * 1024) // 1 MB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, err := dev.allocateDirect(size)
		if err != nil {
			b.Fatalf("Allocation failed: %v", err)
		}
		buf.Free()
	}
}

// BenchmarkCUDAHostToDevice benchmarks host to device transfer
func BenchmarkCUDAHostToDevice(b *testing.B) {
	dev, err := NewCUDADevice()
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	sizes := []int64{
		1024,              // 1 KB
		1024 * 1024,       // 1 MB
		10 * 1024 * 1024,  // 10 MB
		100 * 1024 * 1024, // 100 MB
	}

	for _, size := range sizes {
		b.Run(formatSize(size), func(b *testing.B) {
			buf, err := dev.Allocate(size)
			if err != nil {
				b.Fatalf("Allocation failed: %v", err)
			}
			defer buf.Free()

			data := make([]byte, size)
			for i := range data {
				data[i] = byte(i % 256)
			}

			b.ResetTimer()
			b.SetBytes(size)
			for i := 0; i < b.N; i++ {
				if err := buf.CopyFromHost(data); err != nil {
					b.Fatalf("Copy failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkCUDADeviceToHost benchmarks device to host transfer
func BenchmarkCUDADeviceToHost(b *testing.B) {
	dev, err := NewCUDADevice()
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	sizes := []int64{
		1024,              // 1 KB
		1024 * 1024,       // 1 MB
		10 * 1024 * 1024,  // 10 MB
		100 * 1024 * 1024, // 100 MB
	}

	for _, size := range sizes {
		b.Run(formatSize(size), func(b *testing.B) {
			buf, err := dev.Allocate(size)
			if err != nil {
				b.Fatalf("Allocation failed: %v", err)
			}
			defer buf.Free()

			// Fill buffer with data
			data := make([]byte, size)
			for i := range data {
				data[i] = byte(i % 256)
			}
			buf.CopyFromHost(data)

			result := make([]byte, size)

			b.ResetTimer()
			b.SetBytes(size)
			for i := 0; i < b.N; i++ {
				if err := buf.CopyToHost(result); err != nil {
					b.Fatalf("Copy failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkCUDADeviceToDevice benchmarks device to device copy
func BenchmarkCUDADeviceToDevice(b *testing.B) {
	dev, err := NewCUDADevice()
	if err != nil {
		b.Skipf("CUDA not available: %v", err)
	}
	defer dev.Free()

	sizes := []int64{
		1024,              // 1 KB
		1024 * 1024,       // 1 MB
		10 * 1024 * 1024,  // 10 MB
		100 * 1024 * 1024, // 100 MB
	}

	for _, size := range sizes {
		b.Run(formatSize(size), func(b *testing.B) {
			srcBuf, err := dev.Allocate(size)
			if err != nil {
				b.Fatalf("Source allocation failed: %v", err)
			}
			defer srcBuf.Free()

			dstBuf, err := dev.Allocate(size)
			if err != nil {
				b.Fatalf("Destination allocation failed: %v", err)
			}
			defer dstBuf.Free()

			// Fill source buffer
			data := make([]byte, size)
			srcBuf.CopyFromHost(data)

			b.ResetTimer()
			b.SetBytes(size)
			for i := 0; i < b.N; i++ {
				if err := dev.Copy(dstBuf, srcBuf, size); err != nil {
					b.Fatalf("Copy failed: %v", err)
				}
			}
		})
	}
}

// formatSize formats a size in bytes to a human-readable string
func formatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return "1B"
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	units := []string{"KB", "MB", "GB", "TB"}
	value := bytes / div
	return fmt.Sprintf("%d%s", value, units[exp])
}
