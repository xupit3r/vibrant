#!/usr/bin/env bash
# Vibrant Dependency Checker and Installer
# Checks for and optionally installs required dependencies for llama.cpp integration

set -uo pipefail

# Version
readonly SCRIPT_VERSION="1.0.0"
readonly MIN_GO_VERSION="1.21"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Flags
INSTALL_MODE=false
VERBOSE=false
DRY_RUN=false
QUIET=false

# Counters
DEPS_FOUND=0
DEPS_MISSING=0

# Trap errors
trap 'error_handler $? $LINENO' ERR

error_handler() {
    local exit_code=$1
    local line_number=$2
    echo -e "${RED}✗${NC} Script failed at line $line_number with exit code $exit_code" >&2
    exit "$exit_code"
}

# Usage
usage() {
    cat << EOF
Vibrant Dependency Checker v${SCRIPT_VERSION}

Check for and optionally install dependencies required for Vibrant llama.cpp integration.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -i, --install       Install missing dependencies (requires sudo on Linux)
    -v, --verbose       Show detailed output
    -d, --dry-run       Show what would be installed without installing
    -q, --quiet         Minimal output (errors only)
    -h, --help          Show this help message
    --version           Show version information

EXAMPLES:
    $0                  Check dependencies only
    $0 --install        Check and install missing dependencies
    $0 --dry-run        Show what would be installed

DEPENDENCIES:
    Required: Go ${MIN_GO_VERSION}+, C++ compiler, CMake, Make
    Recommended: Git

EXIT CODES:
    0 - All dependencies satisfied
    1 - Missing dependencies
    2 - Installation failed
    3 - Unsupported platform

EOF
    exit 0
}

# Logging functions
log_info() {
    [ "$QUIET" = false ] && echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    [ "$QUIET" = false ] && echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1" >&2
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

log_verbose() {
    [ "$VERBOSE" = true ] && [ "$QUIET" = false ] && echo -e "  $1"
}

# Detect OS
detect_os() {
    local os_type=""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            os_type="$ID"
        else
            os_type="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_type="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        os_type="windows"
    else
        os_type="unknown"
    fi
    echo "$os_type"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

is_root() {
    [ "${EUID:-$(id -u)}" -eq 0 ]
}

check_go() {
    echo -n "Checking Go... "
    if ! command_exists go; then
        log_error "Not found"
        log_verbose "Install from: https://golang.org/dl/"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
    local go_version major minor
    go_version=$(go version | awk '{print $3}' | sed 's/go//')
    major=$(echo "$go_version" | cut -d. -f1)
    minor=$(echo "$go_version" | cut -d. -f2)
    if [ "$major" -ge 1 ] && [ "$minor" -ge 21 ]; then
        log_success "Go $go_version"
        log_verbose "Location: $(command -v go)"
        DEPS_FOUND=$((DEPS_FOUND + 1))
        return 0
    else
        log_warning "Go $go_version (need ${MIN_GO_VERSION}+)"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
}

check_cpp_compiler() {
    echo -n "Checking C++ compiler... "
    if command_exists g++; then
        local gcc_version
        gcc_version=$(g++ --version | head -n1 | awk '{print $NF}')
        log_success "g++ $gcc_version"
        log_verbose "Location: $(command -v g++)"
        DEPS_FOUND=$((DEPS_FOUND + 1))
        return 0
    elif command_exists clang++; then
        local clang_version
        clang_version=$(clang++ --version | head -n1 | awk '{print $4}')
        log_success "clang++ $clang_version"
        log_verbose "Location: $(command -v clang++)"
        DEPS_FOUND=$((DEPS_FOUND + 1))
        return 0
    else
        log_error "Not found"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
}

check_cmake() {
    echo -n "Checking CMake... "
    if ! command_exists cmake; then
        log_error "Not found"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
    local cmake_version
    cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
    log_success "CMake $cmake_version"
    log_verbose "Location: $(command -v cmake)"
    DEPS_FOUND=$((DEPS_FOUND + 1))
    return 0
}

check_git() {
    echo -n "Checking Git... "
    if ! command_exists git; then
        log_warning "Not found (optional)"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
    local git_version
    git_version=$(git --version | awk '{print $3}')
    log_success "Git $git_version"
    log_verbose "Location: $(command -v git)"
    DEPS_FOUND=$((DEPS_FOUND + 1))
    return 0
}

check_make() {
    echo -n "Checking Make... "
    if ! command_exists make; then
        log_error "Not found"
        DEPS_MISSING=$((DEPS_MISSING + 1))
        return 1
    fi
    local make_version
    make_version=$(make --version | head -n1 | awk '{print $3}')
    log_success "Make $make_version"
    log_verbose "Location: $(command -v make)"
    DEPS_FOUND=$((DEPS_FOUND + 1))
    return 0
}

install_ubuntu() {
    local packages=()
    ! command_exists g++ && packages+=("build-essential")
    ! command_exists cmake && packages+=("cmake")
    ! command_exists git && packages+=("git")
    ! command_exists make && packages+=("make")
    
    [ ${#packages[@]} -eq 0 ] && { log_info "All packages installed"; return 0; }
    
    log_info "Packages to install: ${packages[*]}"
    [ "$DRY_RUN" = true ] && { log_info "Would run: sudo apt-get update && sudo apt-get install -y ${packages[*]}"; return 0; }
    
    ! is_root && ! command_exists sudo && { log_error "sudo not found"; return 2; }
    
    log_info "Updating and installing..."
    if is_root; then
        apt-get update -qq && apt-get install -y -qq "${packages[@]}"
    else
        sudo apt-get update -qq && sudo apt-get install -y -qq "${packages[@]}"
    fi || { log_error "Installation failed"; return 2; }
    
    log_success "Installed ${#packages[@]} package(s)"
}

install_fedora() {
    local packages=()
    ! command_exists g++ && packages+=("gcc-c++")
    ! command_exists cmake && packages+=("cmake")
    ! command_exists git && packages+=("git")
    ! command_exists make && packages+=("make")
    
    [ ${#packages[@]} -eq 0 ] && { log_info "All packages installed"; return 0; }
    
    log_info "Packages to install: ${packages[*]}"
    [ "$DRY_RUN" = true ] && { log_info "Would run: sudo dnf install -y ${packages[*]}"; return 0; }
    
    ! is_root && ! command_exists sudo && { log_error "sudo not found"; return 2; }
    
    log_info "Installing..."
    if is_root; then
        dnf install -y -q "${packages[@]}"
    else
        sudo dnf install -y -q "${packages[@]}"
    fi || { log_error "Installation failed"; return 2; }
    
    log_success "Installed ${#packages[@]} package(s)"
}

install_arch() {
    local packages=()
    ! command_exists g++ && packages+=("base-devel")
    ! command_exists cmake && packages+=("cmake")
    ! command_exists git && packages+=("git")
    
    [ ${#packages[@]} -eq 0 ] && { log_info "All packages installed"; return 0; }
    
    log_info "Packages to install: ${packages[*]}"
    [ "$DRY_RUN" = true ] && { log_info "Would run: sudo pacman -S --noconfirm ${packages[*]}"; return 0; }
    
    ! is_root && ! command_exists sudo && { log_error "sudo not found"; return 2; }
    
    log_info "Installing..."
    if is_root; then
        pacman -S --noconfirm "${packages[@]}"
    else
        sudo pacman -S --noconfirm "${packages[@]}"
    fi || { log_error "Installation failed"; return 2; }
    
    log_success "Installed ${#packages[@]} package(s)"
}

install_macos() {
    if ! xcode-select -p &>/dev/null; then
        log_info "Xcode Command Line Tools not found"
        [ "$DRY_RUN" = true ] && { log_info "Would run: xcode-select --install"; return 0; }
        xcode-select --install 2>/dev/null || true
        log_warning "Complete Xcode installation and run again"
        return 2
    fi
    
    ! command_exists brew && { log_warning "Homebrew not found. Install from: https://brew.sh"; return 2; }
    
    local packages=()
    ! command_exists cmake && packages+=("cmake")
    ! command_exists git && packages+=("git")
    
    [ ${#packages[@]} -eq 0 ] && { log_info "All packages installed"; return 0; }
    
    log_info "Packages to install: ${packages[*]}"
    [ "$DRY_RUN" = true ] && { log_info "Would run: brew install ${packages[*]}"; return 0; }
    
    log_info "Installing..."
    brew install "${packages[@]}" || { log_error "Installation failed"; return 2; }
    
    log_success "Installed ${#packages[@]} package(s)"
}

install_dependencies() {
    local os
    os=$(detect_os)
    log_info "Detected OS: $os"
    echo ""
    
    case $os in
        ubuntu|debian|pop) install_ubuntu ;;
        fedora|rhel|centos|rocky|almalinux) install_fedora ;;
        arch|manjaro) install_arch ;;
        macos) install_macos ;;
        windows)
            log_warning "Windows - manual installation required"
            echo ""
            log_info "See docs/llama-setup.md for Windows setup instructions"
            return 3
            ;;
        *)
            log_error "Unsupported OS: $os"
            log_info "See docs/llama-setup.md for manual installation"
            return 3
            ;;
    esac
}

main() {
    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--install) INSTALL_MODE=true; shift ;;
            -v|--verbose) VERBOSE=true; shift ;;
            -d|--dry-run) DRY_RUN=true; INSTALL_MODE=true; shift ;;
            -q|--quiet) QUIET=true; shift ;;
            --version) echo "v${SCRIPT_VERSION}"; exit 0 ;;
            -h|--help) usage ;;
            *) echo -e "${RED}Unknown option: $1${NC}" >&2; exit 1 ;;
        esac
    done
    
    [ "$QUIET" = false ] && echo "=========================================" && echo "Vibrant Dependency Checker v${SCRIPT_VERSION}" && echo "=========================================" && echo ""
    
    check_go
    check_cpp_compiler
    check_cmake
    check_git
    check_make
    
    [ "$QUIET" = false ] && echo "" && echo "=========================================" && echo "Summary" && echo "=========================================" && echo -e "Found:   ${GREEN}${DEPS_FOUND}${NC}" && echo -e "Missing: ${RED}${DEPS_MISSING}${NC}" && echo ""
    
    if [ "$INSTALL_MODE" = true ] && [ $DEPS_MISSING -gt 0 ]; then
        [ "$DRY_RUN" = true ] && log_info "Dry run mode"
        log_info "Installing missing dependencies..."
        echo ""
        
        install_dependencies
        local result=$?
        
        [ $result -eq 0 ] && echo "" && log_success "Installation complete!" && echo "" && log_info "Verify with: $0" && log_info "Then build: make build" && exit 0
        echo "" && log_error "Installation failed (code: $result)" && exit 2
    elif [ $DEPS_MISSING -gt 0 ]; then
        log_warning "Some dependencies missing"
        echo "" && log_info "To install: $0 --install" && echo "" && log_info "Or see: docs/llama-setup.md"
        exit 1
    else
        log_success "All dependencies satisfied!"
        echo "" && log_info "Build with: make build"
        exit 0
    fi
}

main "$@"
