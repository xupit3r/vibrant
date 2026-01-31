package system

import (
	"testing"
)

func TestGetRAMInfo(t *testing.T) {
	info, err := GetRAMInfo()
	if err != nil {
		t.Fatalf("GetRAMInfo failed: %v", err)
	}
	
	if info.TotalBytes <= 0 {
		t.Errorf("Expected positive total bytes, got %d", info.TotalBytes)
	}
	
	if info.AvailableBytes < 0 {
		t.Errorf("Expected non-negative available bytes, got %d", info.AvailableBytes)
	}
	
	if info.AvailableBytes > info.TotalBytes {
		t.Errorf("Available bytes (%d) cannot exceed total bytes (%d)", 
			info.AvailableBytes, info.TotalBytes)
	}
}

func TestEstimateUsableRAM(t *testing.T) {
	usable, err := EstimateUsableRAM()
	if err != nil {
		t.Fatalf("EstimateUsableRAM failed: %v", err)
	}
	
	if usable < 0 {
		t.Errorf("Expected non-negative usable RAM, got %d", usable)
	}
	
	// Usable RAM should be less than total (due to buffer)
	total, _ := GetTotalRAM()
	if usable > total {
		t.Errorf("Usable RAM (%d) cannot exceed total RAM (%d)", usable, total)
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes    int64
		expected string
	}{
		{0, "0 B"},
		{1023, "1023 B"},
		{1024, "1.0 KiB"},
		{1024 * 1024, "1.0 MiB"},
		{1024 * 1024 * 1024, "1.0 GiB"},
		{1536 * 1024 * 1024, "1.5 GiB"},
	}
	
	for _, tt := range tests {
		result := FormatBytes(tt.bytes)
		if result != tt.expected {
			t.Errorf("FormatBytes(%d) = %s; want %s", tt.bytes, result, tt.expected)
		}
	}
}

func TestGetPlatform(t *testing.T) {
	platform := GetPlatform()
	if platform == "" {
		t.Error("Expected non-empty platform string")
	}
	
	validPlatforms := []string{"linux", "darwin", "windows"}
	valid := false
	for _, p := range validPlatforms {
		if platform == p {
			valid = true
			break
		}
	}
	
	if !valid {
		t.Logf("Warning: Unexpected platform: %s", platform)
	}
}

func TestGetArchitecture(t *testing.T) {
	arch := GetArchitecture()
	if arch == "" {
		t.Error("Expected non-empty architecture string")
	}
	
	validArchs := []string{"amd64", "arm64", "386", "arm"}
	valid := false
	for _, a := range validArchs {
		if arch == a {
			valid = true
			break
		}
	}
	
	if !valid {
		t.Logf("Warning: Unexpected architecture: %s", arch)
	}
}
