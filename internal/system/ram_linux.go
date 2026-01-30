package system

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func getRAMInfo() (*RAMInfo, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, fmt.Errorf("failed to open /proc/meminfo: %w", err)
	}
	defer file.Close()

	var totalKB, availableKB int64
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		
		key := strings.TrimSuffix(fields[0], ":")
		value, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		
		switch key {
		case "MemTotal":
			totalKB = value
		case "MemAvailable":
			availableKB = value
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read /proc/meminfo: %w", err)
	}
	
	if totalKB == 0 {
		return nil, fmt.Errorf("could not determine total RAM")
	}
	
	// Convert KB to bytes
	totalBytes := totalKB * 1024
	availableBytes := availableKB * 1024
	usedBytes := totalBytes - availableBytes
	
	return &RAMInfo{
		TotalBytes:     totalBytes,
		AvailableBytes: availableBytes,
		UsedBytes:      usedBytes,
	}, nil
}
