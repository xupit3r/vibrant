#!/bin/bash
# Test script to verify different build configurations

set -e

echo "========================================="
echo "Vibrant Build Configuration Tests"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
test_build() {
    local name=$1
    local command=$2
    local expected_result=$3
    
    echo -n "Testing $name... "
    
    if eval "$command" > /dev/null 2>&1; then
        if [ "$expected_result" = "pass" ]; then
            echo -e "${GREEN}PASS${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}FAIL${NC} (expected to fail)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    else
        if [ "$expected_result" = "fail" ]; then
            echo -e "${YELLOW}PASS${NC} (expected failure)"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}FAIL${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    fi
}

# Clean before tests
echo "Cleaning build artifacts..."
make clean > /dev/null 2>&1
echo ""

# Test 1: Mock build should always work
echo "1. Mock Engine Build"
test_build "make build-mock" "make build-mock" "pass"
if [ -f "./vibrant" ]; then
    echo "   ✓ Binary created"
    # Test that it's mock
    if ./vibrant ask "test" 2>&1 | grep -q "MOCK\|mock"; then
        echo "   ✓ Mock engine active"
    fi
    rm -f ./vibrant
fi
echo ""

# Test 2: Default build (may be mock or llama)
echo "2. Default Build (auto-detect)"
test_build "make build" "make build" "pass"
if [ -f "./vibrant" ]; then
    echo "   ✓ Binary created"
    # Check which engine
    if ./vibrant ask "test" 2>&1 | grep -q "MOCK\|mock"; then
        echo -e "   ${YELLOW}ℹ${NC} Using mock engine (llama.cpp not available)"
    else
        echo -e "   ${GREEN}✓${NC} Using llama.cpp"
    fi
    rm -f ./vibrant
fi
echo ""

# Test 3: Verify tests run with mock engine
echo "3. Test Suite (Mock Engine)"
test_build "go test ./internal/llm" "go test ./internal/llm -short" "pass"
echo ""

# Test 4: Full test suite
echo "4. Full Test Suite"
test_build "go test ./..." "go test ./... -short" "pass"
echo ""

# Test 5: Check completion command exists
echo "5. Completion Command"
make build > /dev/null 2>&1
if ./vibrant completion bash > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✓ Completion command works"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAIL${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
rm -f ./vibrant
echo ""

# Test 6: Check help output
echo "6. Help Command"
make build > /dev/null 2>&1
if ./vibrant --help > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✓ Help command works"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAIL${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
rm -f ./vibrant
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
