package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// Config represents the application configuration
type Config struct {
	Model     ModelConfig     `mapstructure:"model"`
	Context   ContextConfig   `mapstructure:"context"`
	Assistant AssistantConfig `mapstructure:"assistant"`
	CLI       CLIConfig       `mapstructure:"cli"`
	Logging   LoggingConfig   `mapstructure:"logging"`
}

type ModelConfig struct {
	Default        string `mapstructure:"default"`
	ContextSize    int    `mapstructure:"context_size"`
	Threads        int    `mapstructure:"threads"`
	BatchSize      int    `mapstructure:"batch_size"`
	CacheDir       string `mapstructure:"cache_dir"`
	MaxCacheSizeGB int    `mapstructure:"max_cache_size_gb"`
	AutoDownload   bool   `mapstructure:"auto_download"`
}

type ContextConfig struct {
	MaxTokens int      `mapstructure:"max_tokens"`
	Exclude   []string `mapstructure:"exclude"`
	Include   []string `mapstructure:"include"`
	MaxFiles  int      `mapstructure:"max_files"`
	Strategy  string   `mapstructure:"strategy"`
}

type AssistantConfig struct {
	Style       string  `mapstructure:"style"`
	Temperature float32 `mapstructure:"temperature"`
	MaxTokens   int     `mapstructure:"max_tokens"`
	Stream      bool    `mapstructure:"stream"`
	SaveHistory bool    `mapstructure:"save_history"`
	HistoryDir  string  `mapstructure:"history_dir"`
}

type CLIConfig struct {
	Color           bool              `mapstructure:"color"`
	SyntaxHighlight bool              `mapstructure:"syntax_highlight"`
	ShowTokens      bool              `mapstructure:"show_tokens"`
	Interactive     InteractiveConfig `mapstructure:"interactive"`
}

type InteractiveConfig struct {
	Prompt        string `mapstructure:"prompt"`
	ShowModel     bool   `mapstructure:"show_model"`
	MultilineMode string `mapstructure:"multiline_mode"`
}

type LoggingConfig struct {
	Level      string `mapstructure:"level"`
	File       string `mapstructure:"file"`
	MaxSizeMB  int    `mapstructure:"max_size_mb"`
	MaxBackups int    `mapstructure:"max_backups"`
	Console    bool   `mapstructure:"console"`
}

// DefaultConfig returns configuration with default values
func DefaultConfig() *Config {
	home, _ := os.UserHomeDir()
	vibrantDir := filepath.Join(home, ".vibrant")

	return &Config{
		Model: ModelConfig{
			Default:        "auto",
			ContextSize:    4096,
			Threads:        0,
			BatchSize:      512,
			CacheDir:       filepath.Join(vibrantDir, "models"),
			MaxCacheSizeGB: 50,
			AutoDownload:   true,
		},
		Context: ContextConfig{
			MaxTokens: 3000,
			Exclude: []string{
				"*.log", "*.tmp", ".git/", "node_modules/",
				"vendor/", "__pycache__/", "*.pyc",
			},
			Include:  []string{"*.go", "*.py", "*.js", "*.ts", "README.md"},
			MaxFiles: 50,
			Strategy: "smart",
		},
		Assistant: AssistantConfig{
			Style:       "concise",
			Temperature: 0.2,
			MaxTokens:   1024,
			Stream:      true,
			SaveHistory: true,
			HistoryDir:  filepath.Join(vibrantDir, "sessions"),
		},
		CLI: CLIConfig{
			Color:           true,
			SyntaxHighlight: true,
			ShowTokens:      false,
			Interactive: InteractiveConfig{
				Prompt:        "vibrant> ",
				ShowModel:     true,
				MultilineMode: "alt+enter",
			},
		},
		Logging: LoggingConfig{
			Level:      "info",
			File:       filepath.Join(vibrantDir, "vibrant.log"),
			MaxSizeMB:  10,
			MaxBackups: 3,
			Console:    true,
		},
	}
}

// Load loads configuration from file, environment, and defaults
func Load(cfgFile string) (*Config, error) {
	v := viper.New()

	// Set defaults
	cfg := DefaultConfig()
	setDefaults(v, cfg)

	// Config file setup
	if cfgFile != "" {
		v.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("finding home directory: %w", err)
		}

		v.AddConfigPath(filepath.Join(home, ".vibrant"))
		v.AddConfigPath(".")
		v.SetConfigType("yaml")
		v.SetConfigName("config")
	}

	// Environment variables
	v.SetEnvPrefix("VIBRANT")
	v.AutomaticEnv()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Read config file
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("reading config: %w", err)
		}
		// Config file not found is okay, use defaults
	}

	// Unmarshal into struct
	if err := v.Unmarshal(cfg); err != nil {
		return nil, fmt.Errorf("unmarshaling config: %w", err)
	}

	// Expand paths
	cfg.ExpandPaths()

	// Validate
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validating config: %w", err)
	}

	return cfg, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.Model.ContextSize < 512 || c.Model.ContextSize > 32768 {
		return errors.New("model.context_size must be between 512 and 32768")
	}

	if c.Assistant.Temperature < 0.0 || c.Assistant.Temperature > 1.0 {
		return errors.New("assistant.temperature must be between 0.0 and 1.0")
	}

	validStrategies := []string{"smart", "full", "minimal"}
	if !contains(validStrategies, c.Context.Strategy) {
		return fmt.Errorf("context.strategy must be one of: %v", validStrategies)
	}

	validLevels := []string{"debug", "info", "warn", "error"}
	if !contains(validLevels, c.Logging.Level) {
		return fmt.Errorf("logging.level must be one of: %v", validLevels)
	}

	return nil
}

// ExpandPaths expands ~ and environment variables in paths
func (c *Config) ExpandPaths() {
	c.Model.CacheDir = expandPath(c.Model.CacheDir)
	c.Assistant.HistoryDir = expandPath(c.Assistant.HistoryDir)
	c.Logging.File = expandPath(c.Logging.File)
}

func expandPath(path string) string {
	if strings.HasPrefix(path, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return os.ExpandEnv(path)
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func setDefaults(v *viper.Viper, cfg *Config) {
	v.SetDefault("model.default", cfg.Model.Default)
	v.SetDefault("model.context_size", cfg.Model.ContextSize)
	v.SetDefault("model.threads", cfg.Model.Threads)
	v.SetDefault("model.batch_size", cfg.Model.BatchSize)
	v.SetDefault("model.cache_dir", cfg.Model.CacheDir)
	v.SetDefault("model.max_cache_size_gb", cfg.Model.MaxCacheSizeGB)
	v.SetDefault("model.auto_download", cfg.Model.AutoDownload)

	v.SetDefault("context.max_tokens", cfg.Context.MaxTokens)
	v.SetDefault("context.exclude", cfg.Context.Exclude)
	v.SetDefault("context.include", cfg.Context.Include)
	v.SetDefault("context.max_files", cfg.Context.MaxFiles)
	v.SetDefault("context.strategy", cfg.Context.Strategy)

	v.SetDefault("assistant.style", cfg.Assistant.Style)
	v.SetDefault("assistant.temperature", cfg.Assistant.Temperature)
	v.SetDefault("assistant.max_tokens", cfg.Assistant.MaxTokens)
	v.SetDefault("assistant.stream", cfg.Assistant.Stream)
	v.SetDefault("assistant.save_history", cfg.Assistant.SaveHistory)
	v.SetDefault("assistant.history_dir", cfg.Assistant.HistoryDir)

	v.SetDefault("cli.color", cfg.CLI.Color)
	v.SetDefault("cli.syntax_highlight", cfg.CLI.SyntaxHighlight)
	v.SetDefault("cli.show_tokens", cfg.CLI.ShowTokens)

	v.SetDefault("logging.level", cfg.Logging.Level)
	v.SetDefault("logging.file", cfg.Logging.File)
	v.SetDefault("logging.max_size_mb", cfg.Logging.MaxSizeMB)
	v.SetDefault("logging.max_backups", cfg.Logging.MaxBackups)
	v.SetDefault("logging.console", cfg.Logging.Console)
}
