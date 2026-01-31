package diff

import (
"bufio"
"fmt"
"os"
"path/filepath"
"strings"
)

// DiffLine represents a line in a diff
type DiffLine struct {
Type    string // "add", "remove", "context"
Content string
LineNum int
}

// FileDiff represents changes to a single file
type FileDiff struct {
OriginalPath string
ModifiedPath string
IsNew        bool
IsDeleted    bool
Hunks        []Hunk
}

// Hunk represents a contiguous block of changes
type Hunk struct {
OriginalStart int
OriginalLines int
ModifiedStart int
ModifiedLines int
Lines         []DiffLine
}

// Generate creates a unified diff between old and new content
func Generate(oldContent, newContent, filename string) (*FileDiff, error) {
oldLines := strings.Split(oldContent, "\n")
newLines := strings.Split(newContent, "\n")

diff := &FileDiff{
OriginalPath: filename,
ModifiedPath: filename,
IsNew:        oldContent == "",
IsDeleted:    newContent == "",
}

// Simple line-by-line diff (could be enhanced with Myers algorithm)
hunk := Hunk{
OriginalStart: 1,
ModifiedStart: 1,
}

maxLen := len(oldLines)
if len(newLines) > maxLen {
maxLen = len(newLines)
}

for i := 0; i < maxLen; i++ {
if i < len(oldLines) && i < len(newLines) {
if oldLines[i] == newLines[i] {
hunk.Lines = append(hunk.Lines, DiffLine{
Type:    "context",
Content: oldLines[i],
LineNum: i + 1,
})
hunk.OriginalLines++
hunk.ModifiedLines++
} else {
// Line changed
hunk.Lines = append(hunk.Lines, DiffLine{
Type:    "remove",
Content: oldLines[i],
LineNum: i + 1,
})
hunk.Lines = append(hunk.Lines, DiffLine{
Type:    "add",
Content: newLines[i],
LineNum: i + 1,
})
hunk.OriginalLines++
hunk.ModifiedLines++
}
} else if i < len(oldLines) {
// Line removed
hunk.Lines = append(hunk.Lines, DiffLine{
Type:    "remove",
Content: oldLines[i],
LineNum: i + 1,
})
hunk.OriginalLines++
} else {
// Line added
hunk.Lines = append(hunk.Lines, DiffLine{
Type:    "add",
Content: newLines[i],
LineNum: i + 1,
})
hunk.ModifiedLines++
}
}

if len(hunk.Lines) > 0 {
diff.Hunks = append(diff.Hunks, hunk)
}

return diff, nil
}

// FormatUnified formats a diff in unified diff format
func (d *FileDiff) FormatUnified() string {
var sb strings.Builder

sb.WriteString(fmt.Sprintf("--- a/%s\n", d.OriginalPath))
sb.WriteString(fmt.Sprintf("+++ b/%s\n", d.ModifiedPath))

for _, hunk := range d.Hunks {
sb.WriteString(fmt.Sprintf("@@ -%d,%d +%d,%d @@\n",
hunk.OriginalStart, hunk.OriginalLines,
hunk.ModifiedStart, hunk.ModifiedLines))

for _, line := range hunk.Lines {
switch line.Type {
case "add":
sb.WriteString("+")
case "remove":
sb.WriteString("-")
case "context":
sb.WriteString(" ")
}
sb.WriteString(line.Content)
sb.WriteString("\n")
}
}

return sb.String()
}

// Apply applies a diff to a file
func Apply(diffContent, targetFile string) error {
// Read target file
content, err := os.ReadFile(targetFile)
if err != nil {
return fmt.Errorf("failed to read target file: %w", err)
}

lines := strings.Split(string(content), "\n")
scanner := bufio.NewScanner(strings.NewReader(diffContent))

var newLines []string
lineIdx := 0

for scanner.Scan() {
line := scanner.Text()

if len(line) == 0 {
continue
}

switch line[0] {
case '+':
// Add line
newLines = append(newLines, line[1:])
case '-':
// Skip line (removal)
lineIdx++
case ' ':
// Context line - keep original
if lineIdx < len(lines) {
newLines = append(newLines, lines[lineIdx])
lineIdx++
}
case '@':
// Hunk header - skip
continue
}
}

// Write result
newContent := strings.Join(newLines, "\n")
if err := os.WriteFile(targetFile, []byte(newContent), 0644); err != nil {
return fmt.Errorf("failed to write file: %w", err)
}

return nil
}

// GenerateFromFiles creates a diff between two files
func GenerateFromFiles(originalPath, modifiedPath string) (*FileDiff, error) {
oldContent, err := os.ReadFile(originalPath)
if err != nil && !os.IsNotExist(err) {
return nil, fmt.Errorf("failed to read original file: %w", err)
}

newContent, err := os.ReadFile(modifiedPath)
if err != nil && !os.IsNotExist(err) {
return nil, fmt.Errorf("failed to read modified file: %w", err)
}

filename := filepath.Base(modifiedPath)
return Generate(string(oldContent), string(newContent), filename)
}
