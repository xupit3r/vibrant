package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/xupit3r/vibrant/internal/assistant"
	"github.com/xupit3r/vibrant/internal/llm"
)

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#7B68EE")).
			MarginBottom(1)

	userStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#00D4FF"))

	assistantStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#7FFF00"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF0000")).
			Bold(true)

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666666")).
			Italic(true)

	contextIndicatorStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFA500")).
				Italic(true)
)

type ChatModel struct {
	viewport     viewport.Model
	textarea     textarea.Model
	assistant    *assistant.Assistant
	llmManager   *llm.Manager
	messages     []string
	contextFiles []string
	generating   bool
	err          error
	width        int
	height       int
	ready        bool
}

type streamTokenMsg struct {
	token string
}

type streamDoneMsg struct{}

type streamErrorMsg struct {
	err error
}

func NewChatModel(asst *assistant.Assistant, llmMgr *llm.Manager, contextFiles []string) ChatModel {
	ta := textarea.New()
	ta.Placeholder = "Type your question..."
	ta.Focus()
	ta.Prompt = "❯ "
	ta.CharLimit = 4000
	ta.SetWidth(80)
	ta.SetHeight(3)
	ta.ShowLineNumbers = false

	vp := viewport.New(80, 20)
	vp.SetContent(titleStyle.Render("🤖 Vibrant Chat - Press Ctrl+C to exit, Ctrl+D to clear"))

	return ChatModel{
		viewport:     vp,
		textarea:     ta,
		assistant:    asst,
		llmManager:   llmMgr,
		messages:     []string{},
		contextFiles: contextFiles,
		generating:   false,
		ready:        false,
	}
}

func (m ChatModel) Init() tea.Cmd {
	return textarea.Blink
}

func (m ChatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
	)

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		if !m.ready {
			m.viewport = viewport.New(msg.Width, msg.Height-8)
			m.textarea.SetWidth(msg.Width - 4)
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height - 8
			m.textarea.SetWidth(msg.Width - 4)
		}

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit

		case tea.KeyCtrlD:
			// Clear conversation
			m.messages = []string{}
			m.viewport.SetContent(titleStyle.Render("🤖 Vibrant Chat - Conversation cleared"))
			return m, nil

		case tea.KeyEnter:
			if m.generating {
				return m, nil
			}

			input := strings.TrimSpace(m.textarea.Value())
			if input == "" {
				return m, nil
			}

			m.textarea.Reset()
			m.generating = true

			// Add user message
			m.messages = append(m.messages, userStyle.Render("You: ")+input)
			m.updateViewport()

			// Generate response
			return m, m.generateResponse(input)
		}

	case streamTokenMsg:
		// Append token to last message
		if len(m.messages) > 0 {
			m.messages[len(m.messages)-1] += msg.token
		} else {
			m.messages = append(m.messages, assistantStyle.Render("Assistant: ")+msg.token)
		}
		m.updateViewport()
		return m, nil

	case streamDoneMsg:
		m.generating = false
		m.updateViewport()
		return m, nil

	case streamErrorMsg:
		m.err = msg.err
		m.generating = false
		m.messages = append(m.messages, errorStyle.Render(fmt.Sprintf("Error: %v", msg.err)))
		m.updateViewport()
		return m, nil
	}

	return m, tea.Batch(tiCmd, vpCmd)
}

func (m ChatModel) View() string {
	if !m.ready {
		return "Initializing..."
	}

	var sb strings.Builder

	// Context indicator
	if len(m.contextFiles) > 0 {
		contextInfo := fmt.Sprintf("📁 Context: %d files loaded", len(m.contextFiles))
		sb.WriteString(contextIndicatorStyle.Render(contextInfo))
		sb.WriteString("\n\n")
	}

	// Viewport with messages
	sb.WriteString(m.viewport.View())
	sb.WriteString("\n\n")

	// Input area
	if m.generating {
		sb.WriteString(helpStyle.Render("⏳ Generating response..."))
		sb.WriteString("\n\n")
	}

	sb.WriteString(m.textarea.View())
	sb.WriteString("\n")

	// Help text
	sb.WriteString(helpStyle.Render("Ctrl+C: Exit | Ctrl+D: Clear | Enter: Send"))

	return sb.String()
}

func (m *ChatModel) updateViewport() {
	content := strings.Join(m.messages, "\n\n")
	m.viewport.SetContent(content)
	m.viewport.GotoBottom()
}

func (m *ChatModel) generateResponse(input string) tea.Cmd {
	return func() tea.Msg {
		// Add assistant prefix
		m.messages = append(m.messages, assistantStyle.Render("Assistant: "))

		ctx := context.Background()
		opts := llm.GenerateOptions{
			MaxTokens:   2048,
			Temperature: 0.2,
			TopP:        0.95,
			TopK:        40,
			StopTokens:  []string{},
		}

		// Use streaming
		stream, err := m.llmManager.GenerateStream(ctx, input, opts)
		if err != nil {
			return streamErrorMsg{err: err}
		}

		// Read stream in separate goroutine and send tokens as messages
		go func() {
			for token := range stream {
				// Send token as message to update UI
				tea.NewProgram(m).Send(streamTokenMsg{token: token})
			}
			tea.NewProgram(m).Send(streamDoneMsg{})
		}()

		return nil
	}
}
