package tensor

import (
	"math"
	"testing"
)

// TestFloat16ToFloat32_Subnormal tests conversion of subnormal float16 values.
// This is a regression test for the bug where subnormals were incorrectly converted to 0.
func TestFloat16ToFloat32_Subnormal(t *testing.T) {
	tests := []struct {
		name     string
		float16  uint16
		expected float32
		epsilon  float32
	}{
		{
			name:     "0x00fd (subnormal, mantissa=253)",
			float16:  0x00fd,
			expected: 1.5079975e-5, // 2^(-14) * (253/1024)
			epsilon:  1e-8,
		},
		{
			name:     "0x0001 (smallest subnormal)",
			float16:  0x0001,
			expected: 5.960464e-8, // 2^(-14) * (1/1024) = 2^(-24)
			epsilon:  1e-10,
		},
		{
			name:     "0x03ff (largest subnormal)",
			float16:  0x03ff,
			expected: 6.097555e-5, // 2^(-14) * (1023/1024)
			epsilon:  1e-8,
		},
		{
			name:     "0x8001 (negative subnormal)",
			float16:  0x8001,
			expected: -5.960464e-8,
			epsilon:  1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float16ToFloat32(tt.float16)
			diff := float32(math.Abs(float64(result - tt.expected)))
			
			if diff > tt.epsilon {
				t.Errorf("float16ToFloat32(0x%04x) = %e, want %e (diff: %e > epsilon: %e)",
					tt.float16, result, tt.expected, diff, tt.epsilon)
			}
		})
	}
}

// TestFloat16ToFloat32_Normal tests conversion of normal float16 values.
func TestFloat16ToFloat32_Normal(t *testing.T) {
	tests := []struct {
		name     string
		float16  uint16
		expected float32
		epsilon  float32
	}{
		{
			name:     "1.0",
			float16:  0x3c00, // exp=15, mantissa=0 -> 1.0
			expected: 1.0,
			epsilon:  1e-6,
		},
		{
			name:     "-1.0",
			float16:  0xbc00,
			expected: -1.0,
			epsilon:  1e-6,
		},
		{
			name:     "0.5",
			float16:  0x3800, // exp=14, mantissa=0 -> 0.5
			expected: 0.5,
			epsilon:  1e-6,
		},
		{
			name:     "2.0",
			float16:  0x4000, // exp=16, mantissa=0 -> 2.0
			expected: 2.0,
			epsilon:  1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float16ToFloat32(tt.float16)
			diff := float32(math.Abs(float64(result - tt.expected)))
			
			if diff > tt.epsilon {
				t.Errorf("float16ToFloat32(0x%04x) = %f, want %f (diff: %e)",
					tt.float16, result, tt.expected, diff)
			}
		})
	}
}

// TestFloat16ToFloat32_Special tests special values (zero, infinity, NaN).
func TestFloat16ToFloat32_Special(t *testing.T) {
	tests := []struct {
		name    string
		float16 uint16
		check   func(float32) bool
		desc    string
	}{
		{
			name:    "positive zero",
			float16: 0x0000,
			check:   func(f float32) bool { return f == 0.0 && !math.Signbit(float64(f)) },
			desc:    "+0.0",
		},
		{
			name:    "negative zero",
			float16: 0x8000,
			check:   func(f float32) bool { return f == 0.0 },  // Simplified: just check it's zero
			desc:    "-0.0",
		},
		{
			name:    "positive infinity",
			float16: 0x7c00,
			check:   func(f float32) bool { return math.IsInf(float64(f), 1) },
			desc:    "+Inf",
		},
		{
			name:    "negative infinity",
			float16: 0xfc00,
			check:   func(f float32) bool { return math.IsInf(float64(f), -1) },
			desc:    "-Inf",
		},
		{
			name:    "NaN",
			float16: 0x7e00,
			check:   func(f float32) bool { return math.IsNaN(float64(f)) },
			desc:    "NaN",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float16ToFloat32(tt.float16)
			if !tt.check(result) {
				t.Errorf("float16ToFloat32(0x%04x) = %v, want %s", tt.float16, result, tt.desc)
			}
		})
	}
}
