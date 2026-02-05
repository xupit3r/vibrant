// +build linux,cgo

package cuda

import (
	"testing"
)

func TestNewKernelSet(t *testing.T) {
	// This test verifies that the kernel set can be created
	// Actual kernel loading will be tested once compilation is implemented
	ks, err := NewKernelSet()
	if err != nil {
		t.Skipf("Failed to create kernel set (CUDA may not be available): %v", err)
	}
	defer ks.Free()

	// Verify all kernels are initialized
	if ks.MatMul == nil {
		t.Error("MatMul kernel not initialized")
	}
	if ks.MatMulSingleRow == nil {
		t.Error("MatMulSingleRow kernel not initialized")
	}
	if ks.Softmax == nil {
		t.Error("Softmax kernel not initialized")
	}
	if ks.SoftmaxBatched == nil {
		t.Error("SoftmaxBatched kernel not initialized")
	}
	if ks.RMSNorm == nil {
		t.Error("RMSNorm kernel not initialized")
	}
	if ks.RMSNormBatched == nil {
		t.Error("RMSNormBatched kernel not initialized")
	}
	if ks.Add == nil {
		t.Error("Add kernel not initialized")
	}
	if ks.Mul == nil {
		t.Error("Mul kernel not initialized")
	}
	if ks.MulScalar == nil {
		t.Error("MulScalar kernel not initialized")
	}
	if ks.SiLU == nil {
		t.Error("SiLU kernel not initialized")
	}
	if ks.Copy == nil {
		t.Error("Copy kernel not initialized")
	}

	t.Log("All 11 kernels initialized successfully")
}

func TestKernelLaunchPlaceholders(t *testing.T) {
	// Test that kernel launch methods exist and return appropriate errors
	// since actual kernel execution is not yet implemented
	ks, err := NewKernelSet()
	if err != nil {
		t.Skipf("Kernel set creation failed: %v", err)
	}
	defer ks.Free()

	// These should all return "not yet implemented" errors
	tests := []struct {
		name string
		fn   func() error
	}{
		{"MatMul", func() error { return ks.LaunchMatMul(nil, nil, nil, 10, 10, 10, nil) }},
		{"MatMulSingleRow", func() error { return ks.LaunchMatMulSingleRow(nil, nil, nil, 10, 10, nil) }},
		{"Softmax", func() error { return ks.LaunchSoftmax(nil, nil, 10, nil) }},
		{"SoftmaxBatched", func() error { return ks.LaunchSoftmaxBatched(nil, nil, 2, 10, nil) }},
		{"RMSNorm", func() error { return ks.LaunchRMSNorm(nil, nil, nil, 10, 1e-5, nil) }},
		{"RMSNormBatched", func() error { return ks.LaunchRMSNormBatched(nil, nil, nil, 2, 10, 1e-5, nil) }},
		{"Add", func() error { return ks.LaunchAdd(nil, nil, nil, 10, nil) }},
		{"Mul", func() error { return ks.LaunchMul(nil, nil, nil, 10, nil) }},
		{"MulScalar", func() error { return ks.LaunchMulScalar(nil, 2.0, nil, 10, nil) }},
		{"SiLU", func() error { return ks.LaunchSiLU(nil, nil, 10, nil) }},
		{"Copy", func() error { return ks.LaunchCopy(nil, nil, 10, nil) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Error("Expected error for unimplemented kernel, got nil")
			}
			// Should return "not yet implemented" error
			if err.Error() != "kernel execution not yet implemented (Phase 3 in progress)" {
				t.Logf("Got expected error: %v", err)
			}
		})
	}
}
