package commands

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var deviceInfoCmd = &cobra.Command{
	Use:   "device",
	Short: "Show device information",
	Long: `Display information about the compute device being used.

This command shows which device (CPU, GPU, Metal, CUDA) is selected
and provides system information about available compute resources.`,
	RunE: runDeviceInfo,
}

func init() {
	rootCmd.AddCommand(deviceInfoCmd)
}

func runDeviceInfo(cmd *cobra.Command, args []string) error {
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Vibrant Device Information")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	// Get device flag
	deviceFlag := viper.GetString("device")
	fmt.Printf("Device Flag: %s\n\n", deviceFlag)

	// Try to get device
	dev, err := GetDeviceFromFlag(deviceFlag)
	if err != nil {
		fmt.Printf("❌ Device Error: %v\n\n", err)
		
		// Show available options
		fmt.Println("Available devices:")
		fmt.Println("  • auto  - Auto-detect best device")
		fmt.Println("  • cpu   - Force CPU mode")
		
		if runtime.GOOS == "darwin" {
			fmt.Println("  • gpu   - Generic GPU (uses Metal on macOS)")
			fmt.Println("  • metal - Metal GPU (macOS)")
		} else if runtime.GOOS == "linux" {
			fmt.Println("  • gpu   - Generic GPU (uses CUDA on Linux)")
			fmt.Println("  • cuda  - CUDA GPU (requires NVIDIA GPU + CUDA Toolkit)")
		}
		
		return err
	}

	// Display device info
	fmt.Printf("✅ Device: %s\n", GetDeviceName(dev))
	fmt.Printf("   Type: %s\n", dev.Type())
	fmt.Printf("   Platform: %s/%s\n\n", runtime.GOOS, runtime.GOARCH)

	// Memory info if available
	if dev.Type() == 1 { // GPU
		used, total := dev.MemoryUsage()
		if total > 0 {
			usedGB := float64(used) / (1024 * 1024 * 1024)
			totalGB := float64(total) / (1024 * 1024 * 1024)
			usedPercent := float64(used) / float64(total) * 100
			
			fmt.Printf("GPU Memory:\n")
			fmt.Printf("   Used: %.2f GB / %.2f GB (%.1f%%)\n", usedGB, totalGB, usedPercent)
			fmt.Printf("   Free: %.2f GB\n\n", totalGB-usedGB)
		}
	}

	// System info
	fmt.Println("System Information:")
	fmt.Printf("   OS: %s\n", runtime.GOOS)
	fmt.Printf("   Arch: %s\n", runtime.GOARCH)
	fmt.Printf("   CPUs: %d\n", runtime.NumCPU())
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("   System RAM: ~%.2f GB allocated to process\n", float64(m.Sys)/(1024*1024*1024))
	
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")

	return nil
}
