#!/usr/bin/env bash
# Test script for check-deps.sh
# Tests various scenarios and edge cases

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0
SCRIPT_PATH="../scripts/check-deps.sh"

log_test() {
    echo -e "${YELLOW}TEST:${NC} $1"
}

log_pass() {
    echo -e "${GREEN}✓ PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}✗ FAIL${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Test 1: Help output
test_help() {
    log_test "Help output"
    if $SCRIPT_PATH --help | grep -q "Vibrant Dependency Checker"; then
        log_pass
    else
        log_fail "Help text not found"
    fi
}

# Test 2: Version output
test_version() {
    log_test "Version output"
    if $SCRIPT_PATH --version | grep -q "v1.0.0"; then
        log_pass
    else
        log_fail "Version not found"
    fi
}

# Test 3: Basic check (should pass on dev machine)
test_basic_check() {
    log_test "Basic dependency check"
    if $SCRIPT_PATH --quiet; then
        log_pass
    else
        log_fail "Dependency check failed"
    fi
}

# Test 4: Verbose output
test_verbose() {
    log_test "Verbose output"
    local output
    output=$($SCRIPT_PATH --verbose 2>&1)
    if echo "$output" | grep -q "Location"; then
        log_pass
    else
        log_fail "Verbose output not found"
    fi
}

# Test 5: Dry run
test_dry_run() {
    log_test "Dry run mode"
    # Dry run should never actually install anything
    if $SCRIPT_PATH --dry-run --quiet 2>&1 | grep -q "Would run:"; then
        log_pass
    elif $SCRIPT_PATH --dry-run --quiet; then
        # If all deps satisfied, dry run just exits normally
        log_pass
    else
        log_fail "Dry run mode issue"
    fi
}

# Test 6: Invalid option
test_invalid_option() {
    log_test "Invalid option handling"
    if ! $SCRIPT_PATH --invalid-option 2>/dev/null; then
        log_pass
    else
        log_fail "Should fail with invalid option"
    fi
}

# Test 7: Exit codes
test_exit_codes() {
    log_test "Exit codes"
    $SCRIPT_PATH --quiet
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_pass
    else
        log_fail "Expected exit code 0, got $exit_code"
    fi
}

# Test 8: Output format
test_output_format() {
    log_test "Output format"
    local output
    output=$($SCRIPT_PATH 2>&1)
    if echo "$output" | grep -q "Summary"; then
        log_pass
    else
        log_fail "Summary section not found"
    fi
}

# Test 9: Quiet mode
test_quiet_mode() {
    log_test "Quiet mode"
    local output
    output=$($SCRIPT_PATH --quiet 2>&1)
    # Quiet mode should have minimal output
    if [ $(echo "$output" | wc -l) -lt 10 ]; then
        log_pass
    else
        log_fail "Quiet mode too verbose"
    fi
}

# Test 10: Check for required tools
test_tool_detection() {
    log_test "Tool detection"
    local output
    output=$($SCRIPT_PATH 2>&1)
    if echo "$output" | grep -q "Checking Go" && \
       echo "$output" | grep -q "Checking C++"; then
        log_pass
    else
        log_fail "Tool detection incomplete"
    fi
}

# Main
main() {
    echo "========================================="
    echo "check-deps.sh Test Suite"
    echo "========================================="
    echo ""
    
    # Change to test directory
    cd "$(dirname "$0")"
    
    # Run all tests
    test_help
    test_version
    test_basic_check
    test_verbose
    test_dry_run
    test_invalid_option
    test_exit_codes
    test_output_format
    test_quiet_mode
    test_tool_detection
    
    # Summary
    echo ""
    echo "========================================="
    echo "Test Summary"
    echo "========================================="
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo ""
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}Some tests failed${NC}"
        exit 1
    fi
}

main "$@"
