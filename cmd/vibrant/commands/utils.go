package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// SaveResponse saves a response to a file
func SaveResponse(content, outputPath string) error {
	// If no path specified, create default path
	if outputPath == "" {
		timestamp := time.Now().Format("2006-01-02_15-04-05")
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to get home directory: %w", err)
		}
		outputDir := filepath.Join(homeDir, ".vibrant", "conversations")
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			return fmt.Errorf("failed to create output directory: %w", err)
		}
		outputPath = filepath.Join(outputDir, fmt.Sprintf("response_%s.md", timestamp))
	}

	// Ensure parent directory exists
	parentDir := filepath.Dir(outputPath)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return fmt.Errorf("failed to create parent directory: %w", err)
	}

	// Write file
	if err := os.WriteFile(outputPath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("Response saved to: %s\n", outputPath)
	return nil
}
