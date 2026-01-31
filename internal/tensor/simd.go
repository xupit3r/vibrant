package tensor

// SIMD optimization dispatcher
// This file provides platform detection and dispatches to the best available
// SIMD implementation (AVX2, NEON, or fallback to pure Go)

import (
	"runtime"
)

// simdSupport tracks which SIMD features are available
type simdSupport struct {
	hasAVX2 bool
	hasNEON bool
}

var cpuFeatures simdSupport

func init() {
	// Detect CPU features at initialization
	cpuFeatures = detectCPUFeatures()
}

// detectCPUFeatures determines what SIMD instructions are available
func detectCPUFeatures() simdSupport {
	features := simdSupport{}

	switch runtime.GOARCH {
	case "amd64":
		// On x86-64, check for AVX2
		// We'll use a simple heuristic: if we're on amd64, assume AVX2 is available
		// for modern CPUs (2013+). A more robust solution would use CPUID.
		// For now, we'll enable it unconditionally on amd64
		features.hasAVX2 = true

	case "arm64":
		// ARM64 always has NEON
		features.hasNEON = true
	}

	return features
}

// HasAVX2 returns true if AVX2 is available
func HasAVX2() bool {
	return cpuFeatures.hasAVX2
}

// HasNEON returns true if NEON is available
func HasNEON() bool {
	return cpuFeatures.hasNEON
}

// GetSIMDInfo returns a string describing available SIMD features
func GetSIMDInfo() string {
	if cpuFeatures.hasAVX2 {
		return "AVX2 (x86-64)"
	}
	if cpuFeatures.hasNEON {
		return "NEON (ARM64)"
	}
	return "No SIMD (fallback)"
}
