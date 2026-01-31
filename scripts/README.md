# Scripts

Utility scripts for Vibrant development and deployment.

## check-deps.sh

Automated dependency checker and installer for Vibrant's build requirements.

### Features

- Checks for all required dependencies (Go, C++, CMake, Make, Git)
- Validates Go version (requires 1.21+)
- Multi-platform support (Ubuntu, Fedora, Arch, macOS, Windows)
- Automated installation with `--install` flag
- Dry-run mode to preview changes
- Verbose and quiet modes
- Proper error handling with exit codes

### Usage

```bash
# Check dependencies
./scripts/check-deps.sh

# Check with verbose output
./scripts/check-deps.sh --verbose

# Install missing dependencies
./scripts/check-deps.sh --install

# Preview what would be installed
./scripts/check-deps.sh --dry-run

# Quiet mode (for automation)
./scripts/check-deps.sh --quiet
```

### Make Targets

```bash
# Check dependencies
make check-deps

# Install missing dependencies
make install-deps
```

### Exit Codes

- `0` - All dependencies satisfied
- `1` - Missing dependencies (check mode)
- `2` - Installation failed
- `3` - Unsupported platform

### Platform Support

- **Ubuntu/Debian**: Uses `apt-get`
- **Fedora/RHEL/CentOS**: Uses `dnf`
- **Arch/Manjaro**: Uses `pacman`
- **macOS**: Uses Homebrew and Xcode Command Line Tools
- **Windows**: Provides manual installation guide

### Dependencies Checked

**Required:**
- Go 1.21+
- C++ compiler (g++ or clang++)
- CMake
- Make

**Recommended:**
- Git

### Testing

Run the test suite:

```bash
./test/test-check-deps.sh
```

### See Also

- [docs/llama-setup.md](../docs/llama-setup.md) - Detailed setup guide
- [README.md](../README.md) - Quick start guide
- [Makefile](../Makefile) - Build system integration
