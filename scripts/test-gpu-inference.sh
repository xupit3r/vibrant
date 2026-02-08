#!/bin/bash
# Test script for GPU inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔍 GPU Inference Test Script"
echo "=============================="
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA Toolkit 12.0+"
    exit 1
fi

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers"
    exit 1
fi

echo "✓ CUDA Toolkit found"
echo "✓ NVIDIA drivers found"
echo ""

# Display GPU info
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check if model exists
MODEL_PATH="$HOME/.vibrant/models/qwen2.5-coder-3b-q4.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Please download the model first with: vibrant chat --model qwen2.5-coder-3b-q4"
    exit 1
fi
echo "✓ Model found: $MODEL_PATH"
echo ""

# Build CUDA kernels if needed
if [ ! -f "build/cuda/libvibrant_cuda.so" ]; then
    echo "🔨 Building CUDA kernels..."
    ./scripts/compile-cuda-kernels.sh
    echo ""
fi

# Build test program
echo "🔨 Building GPU inference test..."
export CGO_ENABLED=1
export CGO_LDFLAGS="-L$PROJECT_ROOT/build/cuda -lvibrant_cuda"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/cuda:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

go build -tags cuda -o test_gpu_inference test_gpu_inference.go

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi
echo "✓ Build successful"
echo ""

# Run test with GPU monitoring
echo "🚀 Running GPU inference test..."
echo ""

# Start nvidia-smi monitoring in background
nvidia-smi dmon -s pucvmet -c 100 > gpu_stats.txt 2>&1 &
MONITOR_PID=$!

# Run the test
./test_gpu_inference

# Kill monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "📈 GPU Utilization Summary:"
if [ -f gpu_stats.txt ]; then
    grep -v "^#" gpu_stats.txt | tail -n 20 | head -n 10
    rm -f gpu_stats.txt
fi

echo ""
echo "✅ Test complete!"
