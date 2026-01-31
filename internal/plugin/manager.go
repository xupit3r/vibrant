package plugin

import (
"errors"
"fmt"
"sync"
)

// Plugin represents a plugin that can extend Vibrant's functionality
type Plugin interface {
// Name returns the plugin name
Name() string

// Version returns the plugin version
Version() string

// Initialize initializes the plugin
Initialize(config map[string]interface{}) error

// Execute executes the plugin with given input
Execute(input interface{}) (interface{}, error)

// Shutdown cleans up plugin resources
Shutdown() error
}

// PluginInfo holds metadata about a plugin
type PluginInfo struct {
Name        string
Version     string
Description string
Author      string
Enabled     bool
}

// Manager manages plugin lifecycle
type Manager struct {
plugins map[string]Plugin
info    map[string]PluginInfo
mu      sync.RWMutex
}

// NewManager creates a new plugin manager
func NewManager() *Manager {
return &Manager{
plugins: make(map[string]Plugin),
info:    make(map[string]PluginInfo),
}
}

// Register registers a plugin
func (m *Manager) Register(plugin Plugin, info PluginInfo) error {
m.mu.Lock()
defer m.mu.Unlock()

name := plugin.Name()
if name == "" {
return errors.New("plugin name cannot be empty")
}

if _, exists := m.plugins[name]; exists {
return fmt.Errorf("plugin %s already registered", name)
}

m.plugins[name] = plugin
m.info[name] = info

return nil
}

// Unregister unregisters a plugin
func (m *Manager) Unregister(name string) error {
m.mu.Lock()
defer m.mu.Unlock()

plugin, exists := m.plugins[name]
if !exists {
return fmt.Errorf("plugin %s not found", name)
}

// Shutdown the plugin
if err := plugin.Shutdown(); err != nil {
return fmt.Errorf("failed to shutdown plugin %s: %w", name, err)
}

delete(m.plugins, name)
delete(m.info, name)

return nil
}

// Get retrieves a plugin by name
func (m *Manager) Get(name string) (Plugin, error) {
m.mu.RLock()
defer m.mu.RUnlock()

plugin, exists := m.plugins[name]
if !exists {
return nil, fmt.Errorf("plugin %s not found", name)
}

return plugin, nil
}

// List returns all registered plugins
func (m *Manager) List() []PluginInfo {
m.mu.RLock()
defer m.mu.RUnlock()

list := make([]PluginInfo, 0, len(m.info))
for _, info := range m.info {
list = append(list, info)
}

return list
}

// Execute executes a plugin by name
func (m *Manager) Execute(name string, input interface{}) (interface{}, error) {
m.mu.RLock()
plugin, exists := m.plugins[name]
info := m.info[name]
m.mu.RUnlock()

if !exists {
return nil, fmt.Errorf("plugin %s not found", name)
}

if !info.Enabled {
return nil, fmt.Errorf("plugin %s is disabled", name)
}

return plugin.Execute(input)
}

// Enable enables a plugin
func (m *Manager) Enable(name string) error {
m.mu.Lock()
defer m.mu.Unlock()

info, exists := m.info[name]
if !exists {
return fmt.Errorf("plugin %s not found", name)
}

info.Enabled = true
m.info[name] = info

return nil
}

// Disable disables a plugin
func (m *Manager) Disable(name string) error {
m.mu.Lock()
defer m.mu.Unlock()

info, exists := m.info[name]
if !exists {
return fmt.Errorf("plugin %s not found", name)
}

info.Enabled = false
m.info[name] = info

return nil
}

// ShutdownAll shuts down all plugins
func (m *Manager) ShutdownAll() error {
m.mu.Lock()
defer m.mu.Unlock()

var errs []error

for name, plugin := range m.plugins {
if err := plugin.Shutdown(); err != nil {
errs = append(errs, fmt.Errorf("plugin %s: %w", name, err))
}
}

if len(errs) > 0 {
return fmt.Errorf("failed to shutdown %d plugins: %v", len(errs), errs)
}

return nil
}

// Count returns the number of registered plugins
func (m *Manager) Count() int {
m.mu.RLock()
defer m.mu.RUnlock()
return len(m.plugins)
}

// IsRegistered checks if a plugin is registered
func (m *Manager) IsRegistered(name string) bool {
m.mu.RLock()
defer m.mu.RUnlock()
_, exists := m.plugins[name]
return exists
}
