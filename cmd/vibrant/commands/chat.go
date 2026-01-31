package commands

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"
	"github.com/xupit3r/vibrant/internal/assistant"
	ctxpkg "github.com/xupit3r/vibrant/internal/context"
	"github.com/xupit3r/vibrant/internal/model"
	"github.com/xupit3r/vibrant/internal/tui"
)

var (
	chatContextPath string
	chatModel       string
	chatNoContext   bool
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start interactive chat mode",
	Long: `Start an interactive chat session with the code assistant.

In this mode, you can have multi-turn conversations, include code context,
and get help with your coding tasks.`,
	RunE: runChat,
}

func init() {
	// Flags specific to chat command
	chatCmd.Flags().StringVarP(&chatContextPath, "context", "c", ".", "directory to use as context")
	chatCmd.Flags().StringVar(&chatModel, "model", "auto", "override model selection")
	chatCmd.Flags().BoolVar(&chatNoContext, "no-context", false, "disable automatic context gathering")

	rootCmd.AddCommand(chatCmd)
}

func runChat(cmd *cobra.Command, args []string) error {
	fmt.Println("Initializing Vibrant Chat...")

	// Initialize model manager
	modelMgr, err := model.NewManager("~/.vibrant/models", 50)
	if err != nil {
		return fmt.Errorf("failed to initialize model manager: %w", err)
	}

	// Get or select model
	selectedModel, err := modelMgr.GetOrSelectModel(chatModel)
	if err != nil {
		return fmt.Errorf("failed to select model: %w", err)
	}

	fmt.Printf("Selected model: %s (%s)\n", selectedModel.Name, selectedModel.ID)

	// Check if model is cached
	if !modelMgr.Cache.Has(selectedModel.ID) {
		fmt.Printf("Model not cached. Downloading %s (%d MB)...\n", 
			selectedModel.Name, selectedModel.FileSizeMB)
		
		// Set up progress callback
		modelMgr.Downloader.ProgressFunc = func(downloaded, total int64, speed float64) {
			percent := float64(downloaded) / float64(total) * 100
			speedMB := speed / (1024 * 1024)
			fmt.Printf("\rProgress: %.1f%% - %.2f MB/s   ", percent, speedMB)
		}
		
		if err := modelMgr.EnsureModel(selectedModel.ID); err != nil {
			return fmt.Errorf("failed to download model: %w", err)
		}
		fmt.Println()
	}

	// Build context if enabled
	var contextFiles []string
	var fileIndex *ctxpkg.FileIndex

	if !chatNoContext {
		fmt.Println("Building code context...")
		
		indexer, err := ctxpkg.NewIndexer(chatContextPath, ctxpkg.IndexOptions{
			FollowSymlinks: false,
			MaxFileSize:    1024 * 1024, // 1MB
		})
		if err != nil {
			fmt.Printf("Warning: failed to create indexer: %v\n", err)
		} else {
			fileIndex, err = indexer.Index()
			if err != nil {
				fmt.Printf("Warning: failed to index files: %v\n", err)
			} else {
				files := fileIndex.GetAllFiles()
				contextFiles = make([]string, len(files))
				for i, f := range files {
					contextFiles[i] = f.Path
				}
			}
		}
	}

	// Create assistant
	asst, err := assistant.NewAssistant(modelMgr, assistant.AssistantConfig{
		ModelID:          selectedModel.ID,
		TemplateName:     "default",
		MaxHistory:       20,
		ContextWindow:    4096,
		SaveDir:          "~/.vibrant/sessions",
		AutoSave:         false,
		MaxContextTokens: 3000,
		ContextStrategy:  "smart",
	})
	if err != nil {
		return fmt.Errorf("failed to create assistant: %w", err)
	}

	// Set context index if available
	if fileIndex != nil {
		asst.SetContextIndex(fileIndex)
	}

	// Start TUI
	fmt.Println("Starting chat interface...")
	fmt.Println()

	chatModel := tui.NewChatModel(asst, asst.GetLLMManager(), contextFiles)
	p := tea.NewProgram(chatModel, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		return fmt.Errorf("failed to run chat UI: %w", err)
	}

	return nil
}
