package tui

import (
	"bytes"
	"regexp"
	"strings"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
)

var (
	// Regex to find code blocks in markdown
	codeBlockRegex = regexp.MustCompile("(?s)```(\\w*)\\n(.*?)```")
)

// HighlightCode applies syntax highlighting to code blocks in markdown text
func HighlightCode(text string) string {
	return codeBlockRegex.ReplaceAllStringFunc(text, func(match string) string {
		submatch := codeBlockRegex.FindStringSubmatch(match)
		if len(submatch) < 3 {
			return match
		}

		language := submatch[1]
		code := submatch[2]

		// If no language specified, try to detect
		if language == "" {
			language = "text"
		}

		highlighted := highlightCodeBlock(code, language)
		return "```" + language + "\n" + highlighted + "```"
	})
}

// highlightCodeBlock highlights a single code block
func highlightCodeBlock(code, language string) string {
	// Get lexer for language
	lexer := lexers.Get(language)
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)

	// Use terminal256 formatter for ANSI color output
	formatter := formatters.Get("terminal256")
	if formatter == nil {
		formatter = formatters.Fallback
	}

	// Use monokai style (good for dark terminals)
	style := styles.Get("monokai")
	if style == nil {
		style = styles.Fallback
	}

	// Tokenize and format
	iterator, err := lexer.Tokenise(nil, code)
	if err != nil {
		return code
	}

	var buf bytes.Buffer
	err = formatter.Format(&buf, style, iterator)
	if err != nil {
		return code
	}

	return strings.TrimSuffix(buf.String(), "\n")
}

// StripANSI removes ANSI color codes from text
func StripANSI(text string) string {
	ansiRegex := regexp.MustCompile(`\x1b\[[0-9;]*m`)
	return ansiRegex.ReplaceAllString(text, "")
}

// FormatCodeBlock wraps code with visual markers
func FormatCodeBlock(code, language string) string {
	var sb strings.Builder
	sb.WriteString("┌─ " + language + " ─\n")
	
	lines := strings.Split(code, "\n")
	for _, line := range lines {
		sb.WriteString("│ " + line + "\n")
	}
	
	sb.WriteString("└─\n")
	return sb.String()
}
