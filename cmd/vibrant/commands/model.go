package commands

import (
	"fmt"
	"os"
	"text/tabwriter"
	
	"github.com/spf13/cobra"
	"github.com/xupit3r/vibrant/internal/model"
	"github.com/xupit3r/vibrant/internal/system"
)

var modelCmd = &cobra.Command{
	Use:   "model",
	Short: "Manage models",
	Long:  "Download, list, and manage LLM models",
}

var modelListCmd = &cobra.Command{
	Use:   "list",
	Short: "List available or cached models",
	Long:  "List all available models in the registry or models cached locally",
	RunE:  runModelList,
}

var modelInfoCmd = &cobra.Command{
	Use:   "info [model-id]",
	Short: "Show information about a model",
	Args:  cobra.ExactArgs(1),
	RunE:  runModelInfo,
}

var modelDownloadCmd = &cobra.Command{
	Use:   "download [model-id]",
	Short: "Download a model",
	Args:  cobra.ExactArgs(1),
	RunE:  runModelDownload,
}

var modelRemoveCmd = &cobra.Command{
	Use:   "remove [model-id]",
	Short: "Remove a cached model",
	Args:  cobra.ExactArgs(1),
	RunE:  runModelRemove,
}

var (
	listAll    bool
	listCached bool
)

func init() {
	rootCmd.AddCommand(modelCmd)
	modelCmd.AddCommand(modelListCmd)
	modelCmd.AddCommand(modelInfoCmd)
	modelCmd.AddCommand(modelDownloadCmd)
	modelCmd.AddCommand(modelRemoveCmd)
	
	modelListCmd.Flags().BoolVar(&listAll, "all", false, "List all available models")
	modelListCmd.Flags().BoolVar(&listCached, "cached", false, "List only cached models")
}

func runModelList(cmd *cobra.Command, args []string) error {
	// Get system info
	ramInfo, err := system.GetRAMInfo()
	if err != nil {
		return fmt.Errorf("failed to get RAM info: %w", err)
	}
	
	selector, err := model.NewSelector()
	if err != nil {
		return fmt.Errorf("failed to create selector: %w", err)
	}
	
	fmt.Printf("System RAM: %s total, %s available\n",
		system.FormatBytes(ramInfo.TotalBytes),
		system.FormatBytes(ramInfo.AvailableBytes))
	fmt.Printf("Recommended tier: %s\n\n", selector.GetRecommendedTier())
	
	// Initialize cache manager
	manager, err := model.NewManager("~/.vibrant/models", 50)
	if err != nil {
		return fmt.Errorf("failed to initialize model manager: %w", err)
	}
	
	if listCached {
		return listCachedModels(manager)
	}
	
	return listAvailableModels(manager, selector, listAll)
}

func listAvailableModels(manager *model.Manager, selector *model.Selector, showAll bool) error {
	models := model.ListAll()
	
	if !showAll {
		// Filter by RAM
		models = model.FilterByRAM(selector.GetUsableRAM())
		if len(models) == 0 {
			fmt.Println("No models can fit in available RAM.")
			fmt.Println("Use --all to see all models.")
			return nil
		}
	}
	
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tRAM REQ\tSIZE\tCACHED\tRECOMMENDED")
	fmt.Fprintln(w, "--\t----\t-------\t----\t------\t-----------")
	
	for _, m := range models {
		cached := ""
		if manager.Cache.Has(m.ID) {
			cached = "✓"
		}
		
		recommended := ""
		if m.Recommended {
			recommended = "✓"
		}
		
		ramReq := fmt.Sprintf("%d GB", m.RAMRequiredMB/1024)
		size := fmt.Sprintf("%d MB", m.FileSizeMB)
		
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n",
			m.ID, m.Name, ramReq, size, cached, recommended)
	}
	
	w.Flush()
	
	if !showAll {
	
	fmt.Println()
	fmt.Println("Showing only models that fit in available RAM. Use --all to see all models.")
	}
	
	return nil
}

func listCachedModels(manager *model.Manager) error {
	cached := manager.Cache.List()
	
	if len(cached) == 0 {
		fmt.Println("No models cached.")
		return nil
	}
	
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tSIZE\tDOWNLOADED\tLAST USED\tUSE COUNT")
	fmt.Fprintln(w, "--\t----\t----------\t---------\t---------")
	
	for _, m := range cached {
		size := system.FormatBytes(m.SizeBytes)
		downloaded := m.DownloadedAt.Format("2006-01-02")
		lastUsed := m.LastUsed.Format("2006-01-02")
		
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\n",
			m.ID, size, downloaded, lastUsed, m.UseCount)
	}
	
	w.Flush()
	
	totalSize := manager.Cache.GetTotalSize()
	fmt.Printf("\nTotal cache size: %s\n", system.FormatBytes(totalSize))
	
	return nil
}

func runModelInfo(cmd *cobra.Command, args []string) error {
	modelID := args[0]
	
	m, err := model.GetModelByID(modelID)
	if err != nil {
		return err
	}
	
	fmt.Printf("ID:              %s\n", m.ID)
	fmt.Printf("Name:            %s\n", m.Name)
	fmt.Printf("Family:          %s\n", m.Family)
	fmt.Printf("Parameters:      %s\n", m.Parameters)
	fmt.Printf("Quantization:    %s\n", m.Quantization)
	fmt.Printf("Context Window:  %d tokens\n", m.ContextWindow)
	fmt.Printf("File Size:       %d MB\n", m.FileSizeMB)
	fmt.Printf("RAM Required:    %d MB (~%d GB)\n", m.RAMRequiredMB, m.RAMRequiredMB/1024)
	fmt.Printf("Recommended:     %v\n", m.Recommended)
	fmt.Printf("Description:     %s\n", m.Description)
	fmt.Printf("Tags:            %v\n", m.Tags)
	fmt.Printf("HuggingFace:     %s\n", m.HuggingFaceRepo)
	fmt.Printf("Filename:        %s\n", m.Filename)
	
	// Check if cached
	manager, err := model.NewManager("~/.vibrant/models", 50)
	if err == nil && manager.Cache.Has(modelID) {
		cached, _ := manager.Cache.Get(modelID)
		fmt.Printf("\nCached:          Yes\n")
		fmt.Printf("Path:            %s\n", cached.Path)
		fmt.Printf("Downloaded:      %s\n", cached.DownloadedAt.Format("2006-01-02 15:04:05"))
		fmt.Printf("Last Used:       %s\n", cached.LastUsed.Format("2006-01-02 15:04:05"))
		fmt.Printf("Use Count:       %d\n", cached.UseCount)
	} else {
		fmt.Printf("\nCached:          No\n")
	}
	
	// Check if fits in RAM
	selector, err := model.NewSelector()
	if err == nil {
		canFit, _ := selector.CanFit(modelID)
		if canFit {
			fmt.Printf("Fits in RAM:     Yes\n")
		} else {
			fmt.Printf("Fits in RAM:     No (insufficient RAM)\n")
		}
	}
	
	return nil
}

func runModelDownload(cmd *cobra.Command, args []string) error {
	modelID := args[0]
	
	// Verify model exists
	m, err := model.GetModelByID(modelID)
	if err != nil {
		return err
	}
	
	fmt.Printf("Downloading %s (%d MB)...\n", m.Name, m.FileSizeMB)
	
	// Initialize manager
	manager, err := model.NewManager("~/.vibrant/models", 50)
	if err != nil {
		return fmt.Errorf("failed to initialize model manager: %w", err)
	}
	
	// Set progress callback
	manager.Downloader.ProgressFunc = func(downloaded, total int64, speed float64) {
		percent := float64(downloaded) / float64(total) * 100
		speedMB := speed / (1024 * 1024)
		fmt.Printf("\rProgress: %.1f%% (%s / %s) - %.2f MB/s",
			percent,
			system.FormatBytes(downloaded),
			system.FormatBytes(total),
			speedMB)
	}
	
	// Download
	if err := manager.EnsureModel(modelID); err != nil {
		fmt.Println() // New line after progress
		return fmt.Errorf("download failed: %w", err)
	}
	
	
	fmt.Println()
	fmt.Println("Download complete!")
	return nil
}

func runModelRemove(cmd *cobra.Command, args []string) error {
	modelID := args[0]
	
	manager, err := model.NewManager("~/.vibrant/models", 50)
	if err != nil {
		return fmt.Errorf("failed to initialize model manager: %w", err)
	}
	
	if !manager.Cache.Has(modelID) {
		return fmt.Errorf("model not cached: %s", modelID)
	}
	
	if err := manager.Cache.Remove(modelID); err != nil {
		return fmt.Errorf("failed to remove model: %w", err)
	}
	
	fmt.Printf("Removed model: %s\n", modelID)
	return nil
}
