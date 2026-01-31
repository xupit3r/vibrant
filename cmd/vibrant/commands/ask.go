package commands

import (
	"context"
	"fmt"
	"os"
	"strings"
	
	"github.com/spf13/cobra"
	"github.com/xupit3r/vibrant/internal/llm"
	"github.com/xupit3r/vibrant/internal/model"
	ctxpkg "github.com/xupit3r/vibrant/internal/context"
	"github.com/xupit3r/vibrant/internal/tui"
)

var askCmd = &cobra.Command{
	Use:   "ask [question]",
	Short: "Ask a single coding question",
	Long: `Ask a single coding question and get an immediate response.

This command is useful for quick queries without entering interactive mode.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runAsk,
}

var (
	askContextPath string
	askMaxTokens   int
	askTemperature float64
	askModel       string
	askNoStream    bool
	askNoContext   bool
	askSave        string
)

func init() {
	// Flags specific to ask command
	askCmd.Flags().StringVarP(&askContextPath, "context", "c", ".", "directory or file to use as context")
	askCmd.Flags().IntVar(&askMaxTokens, "max-tokens", 1024, "maximum tokens to generate")
	askCmd.Flags().Float64Var(&askTemperature, "temperature", 0.2, "response randomness (0.0-1.0)")
	askCmd.Flags().StringVar(&askModel, "model", "auto", "model to use (auto for automatic selection)")
	askCmd.Flags().BoolVar(&askNoStream, "no-stream", false, "disable streaming output")
	askCmd.Flags().BoolVar(&askNoContext, "no-context", false, "disable automatic context gathering")
	askCmd.Flags().StringVarP(&askSave, "save", "s", "", "save response to file (default: ~/.vibrant/conversations/response_<timestamp>.md)")

	rootCmd.AddCommand(askCmd)
}

func runAsk(cmd *cobra.Command, args []string) error {
	question := strings.Join(args, " ")
	
	fmt.Println("Initializing Vibrant...")
	
	// Initialize model manager
	modelMgr, err := model.NewManager("~/.vibrant/models", 50)
	if err != nil {
		return fmt.Errorf("failed to initialize model manager: %w", err)
	}
	
	// Get or select model
	selectedModel, err := modelMgr.GetOrSelectModel(askModel)
	if err != nil {
		return fmt.Errorf("failed to select model: %w", err)
	}
	
	fmt.Printf("Using model: %s\n", selectedModel.Name)
	
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
	var codeContext *ctxpkg.Context
	if !askNoContext {
		fmt.Println("Gathering code context...")
		
		// Check if context path exists
		if _, err := os.Stat(askContextPath); err == nil {
			// Create indexer
			indexer, err := ctxpkg.NewIndexer(askContextPath, ctxpkg.DefaultIndexOptions())
			if err != nil {
				fmt.Printf("Warning: Failed to create indexer: %v\n", err)
			} else {
				// Index files
				index, err := indexer.Index()
				if err != nil {
					fmt.Printf("Warning: Failed to index files: %v\n", err)
				} else {
					fmt.Printf("Indexed %d files\n", index.FileCount)
					
					// Build context
					builder := ctxpkg.NewBuilder(index, ctxpkg.DefaultBuilderOptions())
					codeContext, err = builder.Build(question)
					if err != nil {
						fmt.Printf("Warning: Failed to build context: %v\n", err)
					} else {
						fmt.Printf("Context: %d files, ~%d tokens\n", len(codeContext.Files), codeContext.TokenCount)
					}
				}
			}
		}
	}
	
	// Initialize LLM manager
	fmt.Println("Loading model into memory...")
	llmMgr := llm.NewManager(modelMgr)
	defer llmMgr.Close()
	
	if err := llmMgr.LoadModel(selectedModel.ID); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	
	fmt.Println("Generating response...")
	fmt.Println()
	
	// Build prompt with context
	prompt := buildPromptWithContext(question, codeContext)
	
	// Generate response
	ctx := context.Background()
	opts := llm.GenerateOptions{
		MaxTokens:   askMaxTokens,
		Temperature: float32(askTemperature),
		TopP:        0.95,
		TopK:        40,
		StopTokens:  []string{},
	}
	
	if askNoStream {
		response, err := llmMgr.Generate(ctx, prompt, opts)
		if err != nil {
			return fmt.Errorf("generation failed: %w", err)
		}
		// Apply syntax highlighting to response
		highlighted := tui.HighlightCode(response)
		fmt.Println(highlighted)
		
		// Save if requested
		if askSave != "" || cmd.Flags().Changed("save") {
			if err := SaveResponse(response, askSave); err != nil {
				fmt.Printf("Warning: failed to save response: %v\n", err)
			}
		}
	} else {
		stream, err := llmMgr.GenerateStream(ctx, prompt, opts)
		if err != nil {
			return fmt.Errorf("generation failed: %w", err)
		}
		
		var fullResponse strings.Builder
		for token := range stream {
			fmt.Print(token)
			fullResponse.WriteString(token)
			os.Stdout.Sync()
		}
		fmt.Println()
		
		// Save if requested
		if askSave != "" || cmd.Flags().Changed("save") {
			if err := SaveResponse(fullResponse.String(), askSave); err != nil {
				fmt.Printf("Warning: failed to save response: %v\n", err)
			}
		}
	}
	
	return nil
}

func buildPromptWithContext(question string, codeContext *ctxpkg.Context) string {
	var sb strings.Builder
	
	sb.WriteString("You are Vibrant, a helpful coding assistant. Answer the following question concisely and accurately.\n\n")
	
	// Add context if available
	if codeContext != nil && len(codeContext.Files) > 0 {
		sb.WriteString(codeContext.FormatContext())
		sb.WriteString("\n\n")
	}
	
	sb.WriteString(fmt.Sprintf("Question: %s\n\n", question))
	sb.WriteString("Answer:")
	
	return sb.String()
}
