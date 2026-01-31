package diff

import (
"bytes"
"fmt"
"os/exec"
"strings"
)

// GitStatus represents the git repository status
type GitStatus struct {
Modified []string
Added    []string
Deleted  []string
Staged   []string
Branch   string
}

// GetGitStatus returns the current git repository status
func GetGitStatus(repoPath string) (*GitStatus, error) {
status := &GitStatus{}

// Get current branch
cmd := exec.Command("git", "-C", repoPath, "rev-parse", "--abbrev-ref", "HEAD")
output, err := cmd.Output()
if err != nil {
return nil, fmt.Errorf("failed to get branch: %w", err)
}
status.Branch = strings.TrimSpace(string(output))

// Get status
cmd = exec.Command("git", "-C", repoPath, "status", "--porcelain")
output, err = cmd.Output()
if err != nil {
return nil, fmt.Errorf("failed to get status: %w", err)
}

lines := strings.Split(string(output), "\n")
for _, line := range lines {
if len(line) < 4 {
continue
}

statusCode := line[:2]
filename := strings.TrimSpace(line[3:])

switch {
case strings.HasPrefix(statusCode, "M"):
if statusCode[0] == 'M' {
status.Staged = append(status.Staged, filename)
} else {
status.Modified = append(status.Modified, filename)
}
case strings.HasPrefix(statusCode, "A"):
status.Added = append(status.Added, filename)
status.Staged = append(status.Staged, filename)
case strings.HasPrefix(statusCode, "D"):
status.Deleted = append(status.Deleted, filename)
status.Staged = append(status.Staged, filename)
case strings.HasPrefix(statusCode, "?"):
// Untracked
}
}

return status, nil
}

// GetStagedDiff returns the diff of staged changes
func GetStagedDiff(repoPath string) (string, error) {
cmd := exec.Command("git", "-C", repoPath, "diff", "--cached")
output, err := cmd.Output()
if err != nil {
return "", fmt.Errorf("failed to get staged diff: %w", err)
}
return string(output), nil
}

// GetUnstagedDiff returns the diff of unstaged changes
func GetUnstagedDiff(repoPath string) (string, error) {
cmd := exec.Command("git", "-C", repoPath, "diff")
output, err := cmd.Output()
if err != nil {
return "", fmt.Errorf("failed to get unstaged diff: %w", err)
}
return string(output), nil
}

// GenerateCommitMessage generates a commit message based on diffs
func GenerateCommitMessage(diff string, context string) string {
var sb strings.Builder

sb.WriteString("# Suggested commit message based on changes:\n")
sb.WriteString("# Review and edit as needed\n\n")

// Analyze diff to suggest message
lines := strings.Split(diff, "\n")
filesChanged := make(map[string]bool)
addedLines := 0
removedLines := 0

for _, line := range lines {
if strings.HasPrefix(line, "+++") {
parts := strings.Split(line, " ")
if len(parts) > 1 {
filesChanged[parts[1]] = true
}
} else if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
addedLines++
} else if strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "---") {
removedLines++
}
}

// Generate message
if len(filesChanged) == 1 {
for file := range filesChanged {
sb.WriteString(fmt.Sprintf("Update %s\n\n", file))
}
} else {
sb.WriteString(fmt.Sprintf("Update %d files\n\n", len(filesChanged)))
}

sb.WriteString(fmt.Sprintf("- %d lines added\n", addedLines))
sb.WriteString(fmt.Sprintf("- %d lines removed\n", removedLines))

if context != "" {
sb.WriteString(fmt.Sprintf("\nContext: %s\n", context))
}

return sb.String()
}

// IsGitRepo checks if a directory is a git repository
func IsGitRepo(path string) bool {
cmd := exec.Command("git", "-C", path, "rev-parse", "--git-dir")
var out bytes.Buffer
cmd.Stdout = &out
err := cmd.Run()
return err == nil
}
