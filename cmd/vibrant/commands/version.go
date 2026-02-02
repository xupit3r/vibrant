package commands

import (
"fmt"

"github.com/spf13/cobra"
)

var versionCmd = &cobra.Command{
Use:   "version",
Short: "Print version information",
Run: func(cmd *cobra.Command, args []string) {
fmt.Println("Vibrant v0.1.0")
fmt.Println("A local LLM code assistant")
fmt.Println("")
fmt.Println("Build: development")
fmt.Println("Go version: 1.22+")
},
}

func init() {
rootCmd.AddCommand(versionCmd)
}
