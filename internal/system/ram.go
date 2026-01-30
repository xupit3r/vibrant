package system

import (
	"fmt"
	"runtime"
)

// RAMInfo contains information about system memory
type RAMInfo struct {
	TotalBytes     int64
	AvailableBytes int64
	UsedBytes      int64
}

// GetRAMInfo returns information about system RAM
func GetRAMInfo() (*RAMInfo, error) {
	return getRAMInfo()
}

// FormatBytes formats bytes as human-readable string
func FormatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// GetAvailableRAM returns available RAM in bytes
func GetAvailableRAM() (int64, error) {
	info, err := GetRAMInfo()
	if err != nil {
		return 0, err
	}
	return info.AvailableBytes, nil
}

// GetTotalRAM returns total RAM in bytes
func GetTotalRAM() (int64, error) {
	info, err := GetRAMInfo()
	if err != nil {
		return 0, err
	}
	return info.TotalBytes, nil
}

// EstimateUsableRAM returns RAM available for model loading
// (leaves buffer for OS and other processes)
func EstimateUsableRAM() (int64, error) {
	available, err := GetAvailableRAM()
	if err != nil {
		return 0, err
	}
	
	// Leave 2GB buffer for OS and other processes
	buffer := int64(2 * 1024 * 1024 * 1024)
	if available < buffer {
		return 0, nil
	}
	
	return available - buffer, nil
}

// GetPlatform returns the current platform
func GetPlatform() string {
	return runtime.GOOS
}

// GetArchitecture returns the system architecture
func GetArchitecture() string {
	return runtime.GOARCH
}
