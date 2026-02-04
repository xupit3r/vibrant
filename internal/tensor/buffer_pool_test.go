package tensor

import (
	"sync"
	"testing"
)

func TestTensorBufferPool(t *testing.T) {
	pool := &TensorBufferPool{
		pools: make(map[int]*sync.Pool),
	}

	// Test Get and Put
	size := 1024
	buf1 := pool.Get(size)
	if len(buf1) != size {
		t.Errorf("Expected buffer size %d, got %d", size, len(buf1))
	}

	// Write some data
	for i := range buf1 {
		buf1[i] = float32(i)
	}

	// Return to pool
	pool.Put(buf1)

	// Get again - should get zeroed buffer from pool
	buf2 := pool.Get(size)
	if len(buf2) != size {
		t.Errorf("Expected buffer size %d, got %d", size, len(buf2))
	}

	// Check that buffer was zeroed
	for i := range buf2 {
		if buf2[i] != 0 {
			t.Errorf("Expected zeroed buffer, got non-zero value at index %d: %f", i, buf2[i])
			break
		}
	}

	pool.Put(buf2)
}

func TestTensorBufferPoolConcurrent(t *testing.T) {
	// Test concurrent access
	pool := GlobalTensorPool
	size := 512

	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				buf := pool.Get(size)
				if len(buf) != size {
					t.Errorf("Expected buffer size %d, got %d", size, len(buf))
				}
				// Do some work
				for k := range buf {
					buf[k] = float32(k)
				}
				pool.Put(buf)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestGetPutTensor(t *testing.T) {
	pool := GlobalTensorPool

	// Get a pooled tensor
	shape := []int{8, 8}
	tensor := pool.GetTensor(shape, Float32)

	if !tensor.pooled {
		t.Error("Tensor should be marked as pooled")
	}

	data := tensor.Data().([]float32)
	if len(data) != 64 {
		t.Errorf("Expected 64 elements, got %d", len(data))
	}

	// Write some data
	for i := range data {
		data[i] = float32(i)
	}

	// Return tensor to pool
	pool.PutTensor(tensor)

	// Get another tensor - should get zeroed buffer
	tensor2 := pool.GetTensor(shape, Float32)
	data2 := tensor2.Data().([]float32)

	allZero := true
	for i := range data2 {
		if data2[i] != 0 {
			allZero = false
			break
		}
	}

	if !allZero {
		t.Error("Expected zeroed tensor from pool")
	}

	pool.PutTensor(tensor2)
}

func BenchmarkTensorAllocation(b *testing.B) {
	size := 512 * 512

	b.Run("Direct", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			buf := make([]float32, size)
			_ = buf
		}
	})

	b.Run("Pooled", func(b *testing.B) {
		pool := GlobalTensorPool
		for i := 0; i < b.N; i++ {
			buf := pool.Get(size)
			pool.Put(buf)
		}
	})
}
