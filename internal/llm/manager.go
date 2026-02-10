package llm

import (
"context"
"fmt"
"sync"

"github.com/xupit3r/vibrant/internal/model"
)

// Manager handles LLM model loading and inference
type Manager struct {
modelManager *model.Manager
engine       Engine
currentModel *model.ModelInfo
loadOpts     LoadOptions
mu           sync.Mutex
}

// NewManager creates a new LLM manager
func NewManager(modelManager *model.Manager) *Manager {
return &Manager{
modelManager: modelManager,
engine:       nil,
currentModel: nil,
loadOpts:     DefaultLoadOptions(),
}
}

// SetLoadOptions sets model loading options
func (m *Manager) SetLoadOptions(opts LoadOptions) {
m.mu.Lock()
defer m.mu.Unlock()
m.loadOpts = opts
}

// LoadModel loads a model by ID
func (m *Manager) LoadModel(modelID string) error {
m.mu.Lock()
defer m.mu.Unlock()

// Get model info
modelInfo, err := m.modelManager.GetOrSelectModel(modelID)
if err != nil {
return fmt.Errorf("failed to get model: %w", err)
}

// Ensure model is downloaded
if err := m.modelManager.EnsureModel(modelInfo.ID); err != nil {
return fmt.Errorf("failed to ensure model: %w", err)
}

// Get cached model path
cached, err := m.modelManager.Cache.Get(modelInfo.ID)
if err != nil {
return fmt.Errorf("failed to get cached model: %w", err)
}

// Unload existing model
if m.engine != nil {
m.engine.Close()
m.engine = nil
}

// Load new model with custom engine (pure Go)
engine, err := NewCustomEngine(cached.Path, m.loadOpts)
if err != nil {
return fmt.Errorf("failed to load model: %w", err)
}

m.engine = engine
m.currentModel = modelInfo

// Update last used
m.modelManager.Cache.UpdateLastUsed(modelInfo.ID)

return nil
}

// Generate generates text from a prompt
func (m *Manager) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error) {
m.mu.Lock()
engine := m.engine
m.mu.Unlock()

if engine == nil {
return "", fmt.Errorf("no model loaded")
}

return engine.Generate(ctx, prompt, opts)
}

// GenerateStream generates text from a prompt (streaming)
func (m *Manager) GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error) {
m.mu.Lock()
engine := m.engine
m.mu.Unlock()

if engine == nil {
return nil, fmt.Errorf("no model loaded")
}

return engine.GenerateStream(ctx, prompt, opts)
}

// FormatPrompt formats a system+user prompt using the loaded model's chat template.
func (m *Manager) FormatPrompt(system, user string) string {
m.mu.Lock()
engine := m.engine
m.mu.Unlock()

if engine == nil {
// No model loaded, fall back to plain text
if system != "" {
return system + "\n\n" + user
}
return user
}

if ce, ok := engine.(*CustomEngine); ok {
return ce.FormatPrompt(system, user)
}

// Non-custom engine fallback
if system != "" {
return system + "\n\n" + user
}
return user
}

// CurrentModel returns the currently loaded model info
func (m *Manager) CurrentModel() *model.ModelInfo {
m.mu.Lock()
defer m.mu.Unlock()
return m.currentModel
}

// IsLoaded checks if a model is loaded
func (m *Manager) IsLoaded() bool {
m.mu.Lock()
defer m.mu.Unlock()
return m.engine != nil
}

// Unload unloads the current model
func (m *Manager) Unload() error {
m.mu.Lock()
defer m.mu.Unlock()

if m.engine != nil {
if err := m.engine.Close(); err != nil {
return err
}
m.engine = nil
m.currentModel = nil
}

return nil
}

// Close releases all resources
func (m *Manager) Close() error {
return m.Unload()
}
