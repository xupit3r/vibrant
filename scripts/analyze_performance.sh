#!/bin/bash
# Performance analysis script for Vibrant inference

set -e

echo "🔍 Vibrant Inference Performance Analysis"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Step 1: Run tensor benchmarks
echo "📊 Step 1: Tensor Operation Benchmarks"
echo "--------------------------------------"
go test ./internal/tensor -bench=. -benchmem -benchtime=3s -run=^$ > /tmp/tensor_bench.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Tensor benchmarks complete"
    echo ""
    echo "Key operations:"
    grep -E "BenchmarkMatMul|BenchmarkDequantize" /tmp/tensor_bench.txt | head -10
else
    echo -e "${RED}✗${NC} Tensor benchmarks failed"
fi
echo ""

# Step 2: Run transformer benchmarks
echo "📊 Step 2: Transformer Benchmarks"
echo "--------------------------------"
go test ./internal/transformer -bench=. -benchmem -benchtime=3s -run=^$ > /tmp/transformer_bench.txt 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Transformer benchmarks complete"
    cat /tmp/transformer_bench.txt
else
    echo -e "${YELLOW}⚠${NC} Transformer benchmarks: $(cat /tmp/transformer_bench.txt 2>&1 | tail -1)"
fi
echo ""

# Step 3: Check for GGUF model
echo "🔎 Step 3: Looking for test models"
echo "---------------------------------"
MODEL_PATH=""
SEARCH_PATHS=(
    "$HOME/.cache/vibrant/models/*.gguf"
    "$HOME/models/*.gguf"
    "$HOME/Downloads/*.gguf"
    "/tmp/*.gguf"
)

for pattern in "${SEARCH_PATHS[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            MODEL_PATH="$file"
            echo -e "${GREEN}✓${NC} Found model: $MODEL_PATH"
            break 2
        fi
    done
done

if [ -z "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠${NC} No GGUF model found. Skipping end-to-end profiling."
    echo ""
    echo "To run full profiling, download a model:"
    echo "  mkdir -p ~/.cache/vibrant/models"
    echo "  # Download a small Q5_K quantized model (3B recommended for testing)"
    echo ""
else
    # Step 4: Profile end-to-end inference
    echo ""
    echo "📈 Step 4: End-to-End Inference Profiling"
    echo "----------------------------------------"

    # Check if profiler script needs compilation
    if [ ! -f "$PROJECT_ROOT/scripts/profile_inference" ]; then
        echo "Compiling profiler..."
        go build -o scripts/profile_inference scripts/profile_inference.go
    fi

    echo "Running profiler with model: $(basename "$MODEL_PATH")"
    ./scripts/profile_inference \
        -model="$MODEL_PATH" \
        -tokens=20 \
        -cpuprofile=cpu.prof \
        -memprofile=mem.prof \
        -prompt="def fibonacci(n):"

    echo ""
    echo -e "${GREEN}✓${NC} Profiling complete"
fi

# Step 5: Analyze hotspots
echo ""
echo "🔥 Step 5: Identifying Performance Hotspots"
echo "------------------------------------------"

echo ""
echo "📊 MatMul Performance Analysis:"
grep "BenchmarkMatMul" /tmp/tensor_bench.txt | awk '{
    name=$1
    gsub(/Benchmark/, "", name)
    gsub(/-16$/, "", name)
    ns=$3
    allocs=$7
    printf "  %-30s %10s ns/op  %5s allocs/op\n", name, ns, allocs
}'

echo ""
echo "📊 Dequantization Performance:"
grep "BenchmarkDequantize" /tmp/tensor_bench.txt | awk '{
    name=$1
    gsub(/Benchmark/, "", name)
    gsub(/-16$/, "", name)
    ns=$3
    throughput=$4
    printf "  %-30s %10s ns/op  %s\n", name, ns, throughput
}'

# Step 6: Summary and recommendations
echo ""
echo "💡 Performance Recommendations"
echo "=============================="
echo ""

# Check MatMul speedup
NAIVE_TIME=$(grep "BenchmarkMatMulComparison/naive_Ā-16" /tmp/tensor_bench.txt | awk '{print $3}' || echo "0")
PARALLEL_TIME=$(grep "BenchmarkMatMulComparison/parallel_Ā-16" /tmp/tensor_bench.txt | awk '{print $3}' || echo "0")

if [ "$NAIVE_TIME" != "0" ] && [ "$PARALLEL_TIME" != "0" ]; then
    SPEEDUP=$(echo "scale=2; $NAIVE_TIME / $PARALLEL_TIME" | bc)
    echo "  ✓ Current MatMul speedup: ${SPEEDUP}x (parallel vs naive)"
fi

echo ""
echo "  Priority optimizations based on analysis:"
echo "  1. Fused quantized MatMul (eliminate dequant overhead)"
echo "  2. SIMD dequantization (2-4x potential speedup)"
echo "  3. Tensor memory pooling (reduce GC pressure)"
echo "  4. KV-cache preallocation (eliminate decode allocations)"
echo ""

# Check if profiles exist
if [ -f "cpu.prof" ]; then
    echo "🔬 Interactive Analysis:"
    echo "  View CPU profile:    go tool pprof -http=:8080 cpu.prof"
    echo "  View memory profile: go tool pprof -http=:8081 mem.prof"
    echo ""
fi

echo "📝 Full benchmark results saved to:"
echo "  Tensor:      /tmp/tensor_bench.txt"
echo "  Transformer: /tmp/transformer_bench.txt"
echo ""
