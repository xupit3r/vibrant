package plugin

import (
"errors"
"testing"
)

// MockPlugin is a mock plugin for testing
type MockPlugin struct {
name            string
version         string
initCalled      bool
executeCalled   bool
shutdownCalled  bool
shouldFailInit  bool
shouldFailExec  bool
executeResult   interface{}
}

func (m *MockPlugin) Name() string {
return m.name
}

func (m *MockPlugin) Version() string {
return m.version
}

func (m *MockPlugin) Initialize(config map[string]interface{}) error {
m.initCalled = true
if m.shouldFailInit {
return errors.New("init failed")
}
return nil
}

func (m *MockPlugin) Execute(input interface{}) (interface{}, error) {
m.executeCalled = true
if m.shouldFailExec {
return nil, errors.New("execution failed")
}
return m.executeResult, nil
}

func (m *MockPlugin) Shutdown() error {
m.shutdownCalled = true
return nil
}

func TestNewManager(t *testing.T) {
m := NewManager()

if m == nil {
t.Fatal("NewManager returned nil")
}

if m.Count() != 0 {
t.Errorf("Expected 0 plugins, got %d", m.Count())
}
}

func TestRegister(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{
Name:    "test",
Version: "1.0",
Enabled: true,
}

err := m.Register(plugin, info)
if err != nil {
t.Fatalf("Register failed: %v", err)
}

if m.Count() != 1 {
t.Errorf("Expected 1 plugin, got %d", m.Count())
}

if !m.IsRegistered("test") {
t.Error("Plugin should be registered")
}
}

func TestRegisterDuplicate(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: true}

m.Register(plugin, info)

// Try to register again
err := m.Register(plugin, info)
if err == nil {
t.Error("Expected error when registering duplicate plugin")
}
}

func TestRegisterEmptyName(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "", version: "1.0"}
info := PluginInfo{Enabled: true}

err := m.Register(plugin, info)
if err == nil {
t.Error("Expected error when registering plugin with empty name")
}
}

func TestUnregister(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: true}

m.Register(plugin, info)

err := m.Unregister("test")
if err != nil {
t.Fatalf("Unregister failed: %v", err)
}

if m.Count() != 0 {
t.Errorf("Expected 0 plugins after unregister, got %d", m.Count())
}

if !plugin.shutdownCalled {
t.Error("Shutdown should be called when unregistering")
}
}

func TestUnregisterNonExistent(t *testing.T) {
m := NewManager()

err := m.Unregister("nonexistent")
if err == nil {
t.Error("Expected error when unregistering non-existent plugin")
}
}

func TestGet(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: true}

m.Register(plugin, info)

retrieved, err := m.Get("test")
if err != nil {
t.Fatalf("Get failed: %v", err)
}

if retrieved.Name() != "test" {
t.Errorf("Expected plugin name 'test', got '%s'", retrieved.Name())
}
}

func TestGetNonExistent(t *testing.T) {
m := NewManager()

_, err := m.Get("nonexistent")
if err == nil {
t.Error("Expected error when getting non-existent plugin")
}
}

func TestList(t *testing.T) {
m := NewManager()

plugin1 := &MockPlugin{name: "plugin1", version: "1.0"}
plugin2 := &MockPlugin{name: "plugin2", version: "2.0"}

m.Register(plugin1, PluginInfo{Name: "plugin1", Enabled: true})
m.Register(plugin2, PluginInfo{Name: "plugin2", Enabled: true})

list := m.List()

if len(list) != 2 {
t.Errorf("Expected 2 plugins in list, got %d", len(list))
}
}

func TestExecute(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{
name:          "test",
version:       "1.0",
executeResult: "success",
}
info := PluginInfo{Name: "test", Enabled: true}

m.Register(plugin, info)

result, err := m.Execute("test", "input")
if err != nil {
t.Fatalf("Execute failed: %v", err)
}

if result != "success" {
t.Errorf("Expected result 'success', got '%v'", result)
}

if !plugin.executeCalled {
t.Error("Plugin Execute should be called")
}
}

func TestExecuteDisabled(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: false}

m.Register(plugin, info)

_, err := m.Execute("test", "input")
if err == nil {
t.Error("Expected error when executing disabled plugin")
}
}

func TestExecuteNonExistent(t *testing.T) {
m := NewManager()

_, err := m.Execute("nonexistent", "input")
if err == nil {
t.Error("Expected error when executing non-existent plugin")
}
}

func TestEnable(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: false}

m.Register(plugin, info)

err := m.Enable("test")
if err != nil {
t.Fatalf("Enable failed: %v", err)
}

// Should now be able to execute
_, err = m.Execute("test", "input")
if err != nil {
t.Errorf("Should be able to execute after enabling: %v", err)
}
}

func TestDisable(t *testing.T) {
m := NewManager()
plugin := &MockPlugin{name: "test", version: "1.0"}
info := PluginInfo{Name: "test", Enabled: true}

m.Register(plugin, info)

err := m.Disable("test")
if err != nil {
t.Fatalf("Disable failed: %v", err)
}

// Should not be able to execute
_, err = m.Execute("test", "input")
if err == nil {
t.Error("Should not be able to execute after disabling")
}
}

func TestShutdownAll(t *testing.T) {
m := NewManager()

plugin1 := &MockPlugin{name: "plugin1", version: "1.0"}
plugin2 := &MockPlugin{name: "plugin2", version: "2.0"}

m.Register(plugin1, PluginInfo{Name: "plugin1", Enabled: true})
m.Register(plugin2, PluginInfo{Name: "plugin2", Enabled: true})

err := m.ShutdownAll()
if err != nil {
t.Fatalf("ShutdownAll failed: %v", err)
}

if !plugin1.shutdownCalled {
t.Error("Plugin1 Shutdown should be called")
}

if !plugin2.shutdownCalled {
t.Error("Plugin2 Shutdown should be called")
}
}

func TestCount(t *testing.T) {
m := NewManager()

if m.Count() != 0 {
t.Error("Expected 0 plugins initially")
}

plugin := &MockPlugin{name: "test", version: "1.0"}
m.Register(plugin, PluginInfo{Name: "test", Enabled: true})

if m.Count() != 1 {
t.Errorf("Expected 1 plugin, got %d", m.Count())
}
}

func TestIsRegistered(t *testing.T) {
m := NewManager()

if m.IsRegistered("test") {
t.Error("Plugin should not be registered initially")
}

plugin := &MockPlugin{name: "test", version: "1.0"}
m.Register(plugin, PluginInfo{Name: "test", Enabled: true})

if !m.IsRegistered("test") {
t.Error("Plugin should be registered")
}
}

func TestConcurrentAccess(t *testing.T) {
m := NewManager()

// Register plugins concurrently
done := make(chan bool, 10)
for i := 0; i < 10; i++ {
go func(id int) {
plugin := &MockPlugin{name: string(rune('a' + id)), version: "1.0"}
m.Register(plugin, PluginInfo{Name: string(rune('a' + id)), Enabled: true})
done <- true
}(i)
}

// Wait for all goroutines
for i := 0; i < 10; i++ {
<-done
}

if m.Count() != 10 {
t.Errorf("Expected 10 plugins, got %d", m.Count())
}
}
