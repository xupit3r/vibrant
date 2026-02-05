package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/xupit3r/vibrant/internal/gpu"
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

// GetDeviceFromFlag returns the appropriate device based on the --device flag
func GetDeviceFromFlag(deviceFlag string) (gpu.Device, error) {
	deviceFlag = strings.ToLower(strings.TrimSpace(deviceFlag))

	switch deviceFlag {
	case "auto":
		// Auto-detect: GPU if available, otherwise CPU
		return gpu.GetDefaultDevice()

	case "cpu":
		// Force CPU
		return gpu.NewCPUDevice(), nil

	case "gpu":
		// Generic GPU - try to get best available GPU
		dev, err := gpu.GetDevice(gpu.DeviceTypeGPU)
		if err != nil {
			return nil, fmt.Errorf("GPU not available: %w\nUse --device cpu to force CPU mode", err)
		}
		return dev, nil

	case "metal":
		// Explicitly request Metal (macOS only)
		if runtime.GOOS != "darwin" {
			return nil, fmt.Errorf("Metal is only available on macOS")
		}
		dev, err := gpu.NewMetalDevice()
		if err != nil {
			return nil, fmt.Errorf("Metal not available: %w", err)
		}
		return dev, nil

	case "cuda":
		// Explicitly request CUDA (Linux only)
		if runtime.GOOS != "linux" {
			return nil, fmt.Errorf("CUDA is only available on Linux")
		}
		dev, err := gpu.NewCUDADevice()
		if err != nil {
			return nil, fmt.Errorf("CUDA not available: %w\nMake sure CUDA Toolkit is installed and NVIDIA drivers are loaded", err)
		}
		return dev, nil

	default:
		return nil, fmt.Errorf("unknown device: %s\nValid options: auto, cpu, gpu, metal, cuda", deviceFlag)
	}
}

// GetDeviceName returns a human-readable device name with helpful info
func GetDeviceName(dev gpu.Device) string {
	name := dev.Name()
	dtype := dev.Type()

	switch dtype {
	case gpu.DeviceTypeCPU:
		return fmt.Sprintf("%s (CPU mode)", name)
	case gpu.DeviceTypeGPU:
		// Add platform info for GPU
		if runtime.GOOS == "darwin" {
			return fmt.Sprintf("%s (Metal GPU)", name)
		} else if runtime.GOOS == "linux" {
			return fmt.Sprintf("%s (CUDA GPU)", name)
		}
		return fmt.Sprintf("%s (GPU)", name)
	default:
		return name
	}
}
