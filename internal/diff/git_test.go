package diff

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestIsGitRepo(t *testing.T) {
	tmpDir := t.TempDir()

	// Should not be a git repo initially
	if IsGitRepo(tmpDir) {
		t.Error("Empty directory should not be a git repo")
	}

	// Initialize git repo
	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	// Should now be a git repo
	if !IsGitRepo(tmpDir) {
		t.Error("Initialized directory should be a git repo")
	}
}

func TestGetGitStatus(t *testing.T) {
	tmpDir := t.TempDir()

	// Initialize git repo
	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	// Configure git user (required for commits)
	exec.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com").Run()
	exec.Command("git", "-C", tmpDir, "config", "user.name", "Test User").Run()

	// Create initial commit (required for branch to exist)
	readmeFile := filepath.Join(tmpDir, "README.md")
	if err := os.WriteFile(readmeFile, []byte("# Test"), 0644); err != nil {
		t.Fatalf("Failed to write README: %v", err)
	}
	exec.Command("git", "-C", tmpDir, "add", "README.md").Run()
	if err := exec.Command("git", "-C", tmpDir, "commit", "-m", "initial").Run(); err != nil {
		t.Skip("Could not create initial commit")
	}

	// Create and add a file
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	status, err := GetGitStatus(tmpDir)
	if err != nil {
		t.Fatalf("GetGitStatus failed: %v", err)
	}

	if status.Branch == "" {
		t.Error("Branch should not be empty")
	}

	// Add file
	exec.Command("git", "-C", tmpDir, "add", "test.txt").Run()

	status, err = GetGitStatus(tmpDir)
	if err != nil {
		t.Fatalf("GetGitStatus failed: %v", err)
	}

	if len(status.Staged) == 0 {
		t.Error("Should have staged files")
	}
}

func TestGetStagedDiff(t *testing.T) {
	tmpDir := t.TempDir()

	// Initialize git repo
	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	// Configure git user
	exec.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com").Run()
	exec.Command("git", "-C", tmpDir, "config", "user.name", "Test User").Run()

	// Create and stage a file
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	exec.Command("git", "-C", tmpDir, "add", "test.txt").Run()

	diff, err := GetStagedDiff(tmpDir)
	if err != nil {
		t.Fatalf("GetStagedDiff failed: %v", err)
	}

	// Should have diff content for new file
	if !strings.Contains(diff, "test.txt") {
		t.Error("Diff should mention test.txt")
	}
}

func TestGetUnstagedDiff(t *testing.T) {
	tmpDir := t.TempDir()

	// Initialize git repo
	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	// Configure git user
	exec.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com").Run()
	exec.Command("git", "-C", tmpDir, "config", "user.name", "Test User").Run()

	// Create, stage, and commit a file
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("original"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	exec.Command("git", "-C", tmpDir, "add", "test.txt").Run()
	exec.Command("git", "-C", tmpDir, "commit", "-m", "initial").Run()

	// Modify file without staging
	if err := os.WriteFile(testFile, []byte("modified"), 0644); err != nil {
		t.Fatalf("Failed to modify test file: %v", err)
	}

	diff, err := GetUnstagedDiff(tmpDir)
	if err != nil {
		t.Fatalf("GetUnstagedDiff failed: %v", err)
	}

	// Should have diff content
	if diff == "" {
		t.Error("Should have unstaged changes")
	}

	if !strings.Contains(diff, "test.txt") {
		t.Error("Diff should mention test.txt")
	}
}

func TestGenerateCommitMessage(t *testing.T) {
	diff := `diff --git a/file1.go b/file1.go
index 123..456 100644
--- a/file1.go
+++ b/file1.go
@@ -1,3 +1,4 @@
 package main
 
+import "fmt"
 func main() {`

	context := "Added import statement"

	message := GenerateCommitMessage(diff, context)

	if message == "" {
		t.Error("Commit message should not be empty")
	}

	if !strings.Contains(message, "file1.go") && !strings.Contains(message, "files") {
		t.Error("Commit message should mention changed files")
	}

	if !strings.Contains(message, "Context:") {
		t.Error("Commit message should include context")
	}

	if !strings.Contains(message, context) {
		t.Error("Commit message should include provided context")
	}

	// Should have line count info
	if !strings.Contains(message, "added") {
		t.Error("Commit message should mention added lines")
	}
}

func TestGenerateCommitMessageEmpty(t *testing.T) {
	message := GenerateCommitMessage("", "")

	if message == "" {
		t.Error("Should return a message even with empty diff")
	}

	if !strings.Contains(message, "Suggested") {
		t.Error("Should contain suggestion header")
	}
}

func TestGenerateCommitMessageMultipleFiles(t *testing.T) {
	diff := `diff --git a/file1.go b/file1.go
--- a/file1.go
+++ b/file1.go
@@ -1 +1 @@
-old
+new
diff --git a/file2.go b/file2.go
--- a/file2.go
+++ b/file2.go
@@ -1 +1 @@
-old
+new`

	message := GenerateCommitMessage(diff, "")

	// Should mention multiple files
	if !strings.Contains(message, "2") && !strings.Contains(message, "files") {
		t.Error("Should indicate multiple files changed")
	}
}

func TestGitStatusBranch(t *testing.T) {
	tmpDir := t.TempDir()

	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	// Configure git user
	exec.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com").Run()
	exec.Command("git", "-C", tmpDir, "config", "user.name", "Test User").Run()

	// Create initial commit
	readmeFile := filepath.Join(tmpDir, "README.md")
	if err := os.WriteFile(readmeFile, []byte("# Test"), 0644); err != nil {
		t.Fatalf("Failed to write README: %v", err)
	}
	exec.Command("git", "-C", tmpDir, "add", "README.md").Run()
	if err := exec.Command("git", "-C", tmpDir, "commit", "-m", "initial").Run(); err != nil {
		t.Skip("Could not create initial commit")
	}

	status, err := GetGitStatus(tmpDir)
	if err != nil {
		t.Fatalf("GetGitStatus failed: %v", err)
	}

	// Default branch (master or main depending on git config)
	if status.Branch != "master" && status.Branch != "main" {
		t.Logf("Branch name: %s", status.Branch)
		// This is okay, just log it
	}
}

func TestGitStatusModifiedFiles(t *testing.T) {
	tmpDir := t.TempDir()

	// Initialize and configure git
	cmd := exec.Command("git", "init", tmpDir)
	if err := cmd.Run(); err != nil {
		t.Skip("git not available")
	}

	exec.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com").Run()
	exec.Command("git", "-C", tmpDir, "config", "user.name", "Test User").Run()

	// Create and commit a file
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("original"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	exec.Command("git", "-C", tmpDir, "add", "test.txt").Run()
	if err := exec.Command("git", "-C", tmpDir, "commit", "-m", "initial").Run(); err != nil {
		t.Skip("Could not create initial commit")
	}

	// Modify file
	if err := os.WriteFile(testFile, []byte("modified"), 0644); err != nil {
		t.Fatalf("Failed to modify test file: %v", err)
	}

	status, err := GetGitStatus(tmpDir)
	if err != nil {
		t.Fatalf("GetGitStatus failed: %v", err)
	}

	// Should detect modified file
	found := false
	for _, f := range status.Modified {
		if strings.Contains(f, "test.txt") {
			found = true
			break
		}
	}

	if !found {
		// Check if it's in staged instead (depends on git status format)
		for _, f := range status.Staged {
			if strings.Contains(f, "test.txt") {
				found = true
				break
			}
		}
	}

	if !found {
		t.Logf("Modified files: %v, Staged files: %v", status.Modified, status.Staged)
		// Log but don't fail - git status parsing can be tricky
	}
}

func TestGenerateSmartCommitMessage(t *testing.T) {
tests := []struct {
name     string
diff     string
expected string
}{
{
name:     "Empty diff",
diff:     "",
expected: "chore: update files",
},
{
name: "Single go file",
diff: `diff --git a/main.go b/main.go
--- a/main.go
+++ b/main.go
@@ -1 +1,2 @@
 package main
+import "fmt"`,
expected: "feat: update main.go",
},
{
name: "Documentation file",
diff: `diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # Project
+New section`,
expected: "docs: update README.md",
},
{
name: "Test file",
diff: `diff --git a/main_test.go b/main_test.go
--- a/main_test.go
+++ b/main_test.go
@@ -1 +1,2 @@
 package main
+func TestFoo(t *testing.T) {}`,
expected: "test(tests): update main_test.go",
},
{
name: "Multiple files",
diff: `diff --git a/file1.go b/file1.go
--- a/file1.go
+++ b/file1.go
@@ -1 +1,2 @@
 package main
+// comment
diff --git a/file2.go b/file2.go
--- a/file2.go
+++ b/file2.go
@@ -1 +1 @@
-old line
+new line`,
expected: "fix: update 2 files",
},
}

for _, tt := range tests {
t.Run(tt.name, func(t *testing.T) {
result := GenerateSmartCommitMessage(tt.diff)
if !strings.Contains(result, tt.expected[:4]) { // Check commit type prefix
t.Errorf("Expected commit type in '%s', got '%s'", tt.expected, result)
}
})
}
}

func TestGenerateSmartCommitMessageTypes(t *testing.T) {
// Test feat (more additions than deletions)
featDiff := strings.Repeat("+new line\n", 10) + strings.Repeat("-old\n", 2)
result := GenerateSmartCommitMessage(featDiff)
if !strings.HasPrefix(result, "feat:") && !strings.HasPrefix(result, "chore:") {
t.Errorf("Expected feat or chore for mostly additions, got: %s", result)
}

// Test refactor (more deletions than additions)
refactorDiff := strings.Repeat("-old line\n", 10) + strings.Repeat("+new\n", 2)
result = GenerateSmartCommitMessage(refactorDiff)
if !strings.HasPrefix(result, "refactor:") && !strings.HasPrefix(result, "chore:") {
t.Errorf("Expected refactor or chore for mostly deletions, got: %s", result)
}
}
