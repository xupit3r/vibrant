package main

import (
	"github.com/xupit3r/vibrant/cmd/vibrant/commands"
	"os"
)

func main() {
	if err := commands.Execute(); err != nil {
		os.Exit(1)
	}
}
