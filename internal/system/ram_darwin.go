package system

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

func getRAMInfo() (*RAMInfo, error) {
	// Get total memory using sysctl
	totalCmd := exec.Command("sysctl", "-n", "hw.memsize")
	totalOutput, err := totalCmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get total memory: %w", err)
	}
	
	totalBytes, err := strconv.ParseInt(strings.TrimSpace(string(totalOutput)), 10, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse total memory: %w", err)
	}
	
	// Get VM stats for available memory
	vmCmd := exec.Command("vm_stat")
	vmOutput, err := vmCmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get vm_stat: %w", err)
	}
	
	var freePages, inactivePages int64
	pageSize := int64(4096) // Default page size on macOS
	
	lines := strings.Split(string(vmOutput), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "Pages free:") {
			fields := strings.Fields(line)
			if len(fields) >= 3 {
				freePages, _ = strconv.ParseInt(strings.TrimSuffix(fields[2], "."), 10, 64)
			}
		} else if strings.HasPrefix(line, "Pages inactive:") {
			fields := strings.Fields(line)
			if len(fields) >= 3 {
				inactivePages, _ = strconv.ParseInt(strings.TrimSuffix(fields[2], "."), 10, 64)
			}
		} else if strings.HasPrefix(line, "page size of") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				pageSize, _ = strconv.ParseInt(fields[3], 10, 64)
			}
		}
	}
	
	// Available memory is roughly free + inactive pages
	availableBytes := (freePages + inactivePages) * pageSize
	usedBytes := totalBytes - availableBytes
	
	return &RAMInfo{
		TotalBytes:     totalBytes,
		AvailableBytes: availableBytes,
		UsedBytes:      usedBytes,
	}, nil
}
