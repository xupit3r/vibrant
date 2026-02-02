package commands

import (
"fmt"
"os"

"github.com/spf13/cobra"
"github.com/spf13/viper"
)

var (
cfgFile string
verbose bool
quiet   bool
)

// rootCmd represents the base command
var rootCmd = &cobra.Command{
Use:   "vibrant",
Short: "A local LLM code assistant",
Long: `Vibrant is a CPU-optimized local LLM code assistant that provides 
context-aware coding assistance directly in your terminal.

It automatically selects the best model for your system and provides
intelligent coding help without requiring an internet connection.`,
Version: "0.1.0",
}

// Execute runs the root command
func Execute() error {
return rootCmd.Execute()
}

func init() {
cobra.OnInitialize(initConfig)

// Global flags
rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.vibrant/config.yaml)")
rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
rootCmd.PersistentFlags().BoolVarP(&quiet, "quiet", "q", false, "quiet mode")
rootCmd.PersistentFlags().Bool("no-color", false, "disable colored output")

// Bind flags to viper
viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
viper.BindPFlag("quiet", rootCmd.PersistentFlags().Lookup("quiet"))
viper.BindPFlag("no-color", rootCmd.PersistentFlags().Lookup("no-color"))
}

// initConfig reads in config file and ENV variables
func initConfig() {
if cfgFile != "" {
// Use config file from the flag
viper.SetConfigFile(cfgFile)
} else {
// Find home directory
home, err := os.UserHomeDir()
if err != nil {
fmt.Fprintf(os.Stderr, "Error finding home directory: %v\n", err)
os.Exit(1)
}

// Search config in home directory with name ".vibrant"
viper.AddConfigPath(home + "/.vibrant")
viper.AddConfigPath(".")
viper.SetConfigType("yaml")
viper.SetConfigName("config")
}

// Environment variables
viper.SetEnvPrefix("VIBRANT")
viper.AutomaticEnv()

// If a config file is found, read it in
if err := viper.ReadInConfig(); err == nil {
if verbose {
fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
}
}
}
