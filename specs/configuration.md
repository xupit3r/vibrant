# Configuration Specification

## Overview
Vibrant uses a hierarchical configuration system with defaults, config files, environment variables, and CLI flags.

## Configuration Hierarchy (lowest to highest priority)

1. **Hard-coded defaults** (in code)
2. **Config file** (~/.vibrant/config.yaml)
3. **Environment variables** (VIBRANT_*)
4. **CLI flags** (--flag)

## Configuration File Format

### Location
- **Primary**: `~/.vibrant/config.yaml`
- **Project-level**: `.vibrant.yaml` (in project root)
- **Custom**: Via `--config` flag or `VIBRANT_CONFIG` env var

### Schema (YAML)

```yaml
# Model Configuration
model:
  # Auto-select based on RAM, or specify a model ID
  default: "auto"  # or "qwen2.5-coder-7b-q5"
  
  # Model loading options
  context_size: 4096
  threads: 0  # 0 = auto-detect (NumCPU)
  batch_size: 512
  
  # Cache settings
  cache_dir: "~/.vibrant/models"
  max_cache_size_gb: 50
  auto_download: true

# Context System Configuration
context:
  # Maximum tokens to include in context
  max_tokens: 3000
  
  # File patterns to exclude (in addition to .gitignore)
  exclude:
    - "*.log"
    - "*.tmp"
    - "*_test.go"  # Optionally exclude tests
    - "vendor/"
    - "node_modules/"
  
  # Include patterns (override excludes)
  include:
    - "README.md"
    - "*.go"
    - "*.py"
  
  # Maximum files to include
  max_files: 50
  
  # Context building strategy
  strategy: "smart"  # smart, full, minimal

# Assistant Behavior
assistant:
  # Response style
  style: "concise"  # concise, detailed, conversational
  
  # Temperature (0.0-1.0)
  temperature: 0.2
  
  # Maximum tokens to generate
  max_tokens: 1024
  
  # Streaming output
  stream: true
  
  # Save conversation history
  save_history: true
  history_dir: "~/.vibrant/sessions"

# CLI Interface
cli:
  # Color output
  color: true
  
  # Syntax highlighting
  syntax_highlight: true
  
  # Show token count in responses
  show_tokens: false
  
  # Interactive mode settings
  interactive:
    # Prompt style
    prompt: "vibrant> "
    
    # Show model name in prompt
    show_model: true
    
    # Keyboard shortcuts
    multiline_mode: "alt+enter"

# Logging
logging:
  # Log level: debug, info, warn, error
  level: "info"
  
  # Log file location
  file: "~/.vibrant/vibrant.log"
  
  # Max log file size before rotation
  max_size_mb: 10
  
  # Number of old log files to keep
  max_backups: 3
  
  # Console output
  console: true

# Updates (future)
updates:
  check: true
  frequency: "weekly"
  auto_update: false
```

## Environment Variables

All configuration values can be overridden with environment variables using the prefix `VIBRANT_` and uppercase with underscores:

```bash
# Model settings
VIBRANT_MODEL_DEFAULT=qwen2.5-coder-7b-q5
VIBRANT_MODEL_CONTEXT_SIZE=8192
VIBRANT_MODEL_THREADS=8

# Context settings
VIBRANT_CONTEXT_MAX_TOKENS=5000
VIBRANT_CONTEXT_STRATEGY=full

# Assistant settings
VIBRANT_ASSISTANT_TEMPERATURE=0.3
VIBRANT_ASSISTANT_MAX_TOKENS=2048

# Logging
VIBRANT_LOGGING_LEVEL=debug
VIBRANT_LOGGING_FILE=/tmp/vibrant.log

# Paths
VIBRANT_MODEL_CACHE_DIR=/mnt/models/vibrant
VIBRANT_CONFIG=/path/to/custom/config.yaml
```

## CLI Flags

Common flags available across commands:

```bash
# Global flags
--config <path>           # Custom config file
--model <model-id>        # Override model selection
--verbose, -v             # Increase verbosity
--quiet, -q               # Suppress output
--no-color                # Disable colored output

# Context flags (ask/chat commands)
--context <path>          # Add directory/file to context
--max-context <int>       # Max context tokens
--no-context              # Disable automatic context

# Model flags
--temperature <float>     # Response randomness (0.0-1.0)
--max-tokens <int>        # Max tokens to generate
--threads <int>           # CPU threads for inference

# Output flags
--output <file>           # Save response to file
--no-stream               # Disable streaming output
--json                    # JSON output format
```

## Configuration Struct (Go)

```go
type Config struct {
    Model      ModelConfig      `yaml:"model" mapstructure:"model"`
    Context    ContextConfig    `yaml:"context" mapstructure:"context"`
    Assistant  AssistantConfig  `yaml:"assistant" mapstructure:"assistant"`
    CLI        CLIConfig        `yaml:"cli" mapstructure:"cli"`
    Logging    LoggingConfig    `yaml:"logging" mapstructure:"logging"`
    Updates    UpdatesConfig    `yaml:"updates" mapstructure:"updates"`
}

type ModelConfig struct {
    Default          string   `yaml:"default" mapstructure:"default"`
    ContextSize      int      `yaml:"context_size" mapstructure:"context_size"`
    Threads          int      `yaml:"threads" mapstructure:"threads"`
    BatchSize        int      `yaml:"batch_size" mapstructure:"batch_size"`
    CacheDir         string   `yaml:"cache_dir" mapstructure:"cache_dir"`
    MaxCacheSizeGB   int      `yaml:"max_cache_size_gb" mapstructure:"max_cache_size_gb"`
    AutoDownload     bool     `yaml:"auto_download" mapstructure:"auto_download"`
}

type ContextConfig struct {
    MaxTokens    int      `yaml:"max_tokens" mapstructure:"max_tokens"`
    Exclude      []string `yaml:"exclude" mapstructure:"exclude"`
    Include      []string `yaml:"include" mapstructure:"include"`
    MaxFiles     int      `yaml:"max_files" mapstructure:"max_files"`
    Strategy     string   `yaml:"strategy" mapstructure:"strategy"`
}

type AssistantConfig struct {
    Style         string  `yaml:"style" mapstructure:"style"`
    Temperature   float32 `yaml:"temperature" mapstructure:"temperature"`
    MaxTokens     int     `yaml:"max_tokens" mapstructure:"max_tokens"`
    Stream        bool    `yaml:"stream" mapstructure:"stream"`
    SaveHistory   bool    `yaml:"save_history" mapstructure:"save_history"`
    HistoryDir    string  `yaml:"history_dir" mapstructure:"history_dir"`
}

type CLIConfig struct {
    Color           bool              `yaml:"color" mapstructure:"color"`
    SyntaxHighlight bool              `yaml:"syntax_highlight" mapstructure:"syntax_highlight"`
    ShowTokens      bool              `yaml:"show_tokens" mapstructure:"show_tokens"`
    Interactive     InteractiveConfig `yaml:"interactive" mapstructure:"interactive"`
}

type InteractiveConfig struct {
    Prompt          string `yaml:"prompt" mapstructure:"prompt"`
    ShowModel       bool   `yaml:"show_model" mapstructure:"show_model"`
    MultilineMode   string `yaml:"multiline_mode" mapstructure:"multiline_mode"`
}

type LoggingConfig struct {
    Level       string `yaml:"level" mapstructure:"level"`
    File        string `yaml:"file" mapstructure:"file"`
    MaxSizeMB   int    `yaml:"max_size_mb" mapstructure:"max_size_mb"`
    MaxBackups  int    `yaml:"max_backups" mapstructure:"max_backups"`
    Console     bool   `yaml:"console" mapstructure:"console"`
}

type UpdatesConfig struct {
    Check       bool   `yaml:"check" mapstructure:"check"`
    Frequency   string `yaml:"frequency" mapstructure:"frequency"`
    AutoUpdate  bool   `yaml:"auto_update" mapstructure:"auto_update"`
}
```

## Default Values

```go
func DefaultConfig() *Config {
    return &Config{
        Model: ModelConfig{
            Default:         "auto",
            ContextSize:     4096,
            Threads:         0, // auto-detect
            BatchSize:       512,
            CacheDir:        "~/.vibrant/models",
            MaxCacheSizeGB:  50,
            AutoDownload:    true,
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
            HistoryDir:  "~/.vibrant/sessions",
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
            File:       "~/.vibrant/vibrant.log",
            MaxSizeMB:  10,
            MaxBackups: 3,
            Console:    true,
        },
        Updates: UpdatesConfig{
            Check:      true,
            Frequency:  "weekly",
            AutoUpdate: false,
        },
    }
}
```

## Configuration Loading

```go
type ConfigLoader struct {
    viper *viper.Viper
}

func NewConfigLoader() *ConfigLoader {
    v := viper.New()
    
    // Set defaults
    setDefaults(v)
    
    // Config file settings
    v.SetConfigName("config")
    v.SetConfigType("yaml")
    v.AddConfigPath("$HOME/.vibrant")
    v.AddConfigPath(".")  // Current directory
    
    // Environment variables
    v.SetEnvPrefix("VIBRANT")
    v.AutomaticEnv()
    v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
    
    return &ConfigLoader{viper: v}
}

func (l *ConfigLoader) Load() (*Config, error) {
    // Read config file (if exists)
    if err := l.viper.ReadInConfig(); err != nil {
        if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
            return nil, err
        }
    }
    
    // Unmarshal into struct
    var config Config
    if err := l.viper.Unmarshal(&config); err != nil {
        return nil, err
    }
    
    // Validate
    if err := config.Validate(); err != nil {
        return nil, err
    }
    
    // Expand paths
    config.ExpandPaths()
    
    return &config, nil
}
```

## Validation

```go
func (c *Config) Validate() error {
    // Model validation
    if c.Model.ContextSize < 512 || c.Model.ContextSize > 32768 {
        return errors.New("context_size must be between 512 and 32768")
    }
    
    // Temperature validation
    if c.Assistant.Temperature < 0.0 || c.Assistant.Temperature > 1.0 {
        return errors.New("temperature must be between 0.0 and 1.0")
    }
    
    // Strategy validation
    validStrategies := []string{"smart", "full", "minimal"}
    if !contains(validStrategies, c.Context.Strategy) {
        return errors.New("invalid context strategy")
    }
    
    // Log level validation
    validLevels := []string{"debug", "info", "warn", "error"}
    if !contains(validLevels, c.Logging.Level) {
        return errors.New("invalid logging level")
    }
    
    return nil
}
```

## Configuration Commands

```bash
# Show current configuration
vibrant config show

# Get a specific value
vibrant config get model.default

# Set a value
vibrant config set model.default qwen2.5-coder-7b-q5

# Reset to defaults
vibrant config reset

# Validate configuration
vibrant config validate

# Show configuration file location
vibrant config path
```

## Status

- **Current**: Specification phase
- **Last Updated**: 2026-01-30
- **Dependencies**: viper (configuration management)
- **Implementation**: Phase 1 of PLAN.md
