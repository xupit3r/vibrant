#!/bin/bash
# Compile CUDA kernels for Vibrant
# This script compiles kernels.cu and launch.cu into a shared library
# that can be linked with the Go binary

set -e

CUDA_DIR="${CUDA_DIR:-/usr/local/cuda}"
NVCC="${CUDA_DIR}/bin/nvcc"
KERNELS_DIR="internal/gpu/cuda"
BUILD_DIR="build/cuda"

# Check if CUDA toolkit is available
if [ ! -f "$NVCC" ]; then
    echo "❌ CUDA toolkit not found at ${CUDA_DIR}"
    echo "   Set CUDA_DIR environment variable to your CUDA installation path"
    echo "   Example: export CUDA_DIR=/usr/local/cuda-12.0"
    exit 1
fi

echo "🔨 Compiling CUDA kernels..."
echo "   CUDA Toolkit: ${CUDA_DIR}"
echo "   nvcc: ${NVCC}"

# Create build directory
mkdir -p ${BUILD_DIR}

# Compile kernels.cu and launch.cu to object files
echo "   Compiling kernels.cu..."
${NVCC} -c ${KERNELS_DIR}/kernels.cu \
    -o ${BUILD_DIR}/kernels.o \
    -arch=sm_86 \
    -O3 \
    --compiler-options '-fPIC' \
    -I${KERNELS_DIR}

echo "   Compiling launch.cu..."
${NVCC} -c ${KERNELS_DIR}/launch.cu \
    -o ${BUILD_DIR}/launch.o \
    -arch=sm_86 \
    -O3 \
    --compiler-options '-fPIC' \
    -I${KERNELS_DIR}

# Link into shared library
echo "   Linking shared library..."
${NVCC} -shared ${BUILD_DIR}/kernels.o ${BUILD_DIR}/launch.o \
    -o ${BUILD_DIR}/libvibrant_cuda.so \
    -lcudart

echo "✅ CUDA kernels compiled successfully!"
echo "   Output: ${BUILD_DIR}/libvibrant_cuda.so"
echo ""
echo "💡 To build Vibrant with CUDA support:"
echo "   export CGO_LDFLAGS=\"-L$(pwd)/${BUILD_DIR} -lvibrant_cuda\""
echo "   export LD_LIBRARY_PATH=\"$(pwd)/${BUILD_DIR}:\${LD_LIBRARY_PATH}\""
echo "   make build-cuda"
