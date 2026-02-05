# CUDA Setup Guide

This guide explains how to set up Vibrant with CUDA GPU acceleration on Linux.

## Requirements

### Hardware

- **NVIDIA GPU**: RTX 30/40 series recommended (compute capability 8.6+)
  - RTX 3060 Ti / 3070 / 3080 / 3090
  - RTX 4060 / 4070 / 4080 / 4090
  - Or any NVIDIA GPU with CUDA support (compute capability 6.0+)

### Software

- **Operating System**: Linux (Ubuntu 22.04+ recommended)
- **CUDA Toolkit**: Version 12.0 or newer
- **NVIDIA Driver**: Version 525.60.13 or newer
- **Go**: Version 1.21 or newer
- **GCC/G++**: For CGO compilation

## Installation Steps

### 1. Install NVIDIA Drivers

#### Ubuntu/Debian

```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Install NVIDIA driver
sudo apt update
sudo apt install nvidia-driver-535

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

**Expected output**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05     CUDA Version: 12.2   |
|-------------------------------+----------------------+--------------------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+================================|
|   0  NVIDIA GeForce RTX 4090  Off | 00000000:01:00.0  On |                  Off |
| 30%   45C    P8              25W / 450W |    523MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+--------------------------------+
```

### 2. Install CUDA Toolkit

#### Ubuntu/Debian

```bash
# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

**Expected output**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

### 3. Build Vibrant with CUDA

```bash
# Clone repository (if not already done)
git clone https://github.com/your-username/vibrant.git
cd vibrant

# Compile CUDA kernels
./scripts/compile-cuda-kernels.sh

# Build Vibrant with CUDA support
make build-cuda

# Verify the build
./vibrant device --device cuda
```

**Expected output**:
```
═══════════════════════════════════════════════════════
  Vibrant Device Information
═══════════════════════════════════════════════════════

Device Flag: cuda

✅ Device: NVIDIA GeForce RTX 4090 (CUDA GPU)
   Type: GPU
   Platform: linux/amd64

GPU Memory:
   Used: 0.52 GB / 24.00 GB (2.2%)
   Free: 23.48 GB

System Information:
   OS: linux
   Arch: amd64
   CPUs: 24
   System RAM: ~0.12 GB allocated to process
```

## Troubleshooting

### Issue: CUDA not available

**Error**:
```
❌ Device Error: CUDA not available: CUDA not available: No such file or directory
```

**Solution**:
1. Verify CUDA toolkit is installed: `nvcc --version`
2. Check `LD_LIBRARY_PATH` includes CUDA: `echo $LD_LIBRARY_PATH`
3. Rebuild with CUDA: `make clean && make build-cuda`

### Issue: No CUDA devices found

**Error**:
```
❌ Device Error: no CUDA devices found
```

**Solution**:
1. Check GPU is detected: `nvidia-smi`
2. Verify driver is loaded: `lsmod | grep nvidia`
3. Reload driver: `sudo modprobe nvidia`

### Issue: Kernel launch failed

**Error**:
```
CUDA MatMul: kernel launch failed: invalid argument
```

**Solution**:
1. Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
2. Verify kernel was compiled for correct architecture (sm_86)
3. Recompile kernels: `./scripts/compile-cuda-kernels.sh`

### Issue: Out of memory

**Error**:
```
CUDA MatMul: failed to allocate CUDA buffer: out of memory
```

**Solution**:
1. Check available GPU memory: `nvidia-smi`
2. Use a smaller model
3. Reduce batch size
4. Close other GPU-using applications

### Issue: Build fails with "cuda_runtime.h: No such file or directory"

**Solution**:
```bash
# Set CUDA_DIR explicitly
export CUDA_DIR=/usr/local/cuda-12.3
./scripts/compile-cuda-kernels.sh

# Or install CUDA toolkit if not present
sudo apt-get install cuda-toolkit-12-3
```

## Environment Variables

### CUDA_DIR
Path to CUDA installation (default: `/usr/local/cuda`)

```bash
export CUDA_DIR=/usr/local/cuda-12.3
```

### LD_LIBRARY_PATH
Must include CUDA libraries and Vibrant's compiled kernels

```bash
export LD_LIBRARY_PATH=$(pwd)/build/cuda:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA_VISIBLE_DEVICES
Restrict which GPUs are visible to CUDA

```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2
```

## Performance Tuning

### Kernel Configuration

Edit `internal/gpu/cuda/kernels.h` to adjust parameters:

```c
// Tile size for matmul (default: 16)
#define TILE_SIZE 16  // Try 32 for larger matrices

// Block size for element-wise ops (default: 256)
#define BLOCK_SIZE 256  // Try 512 for more parallelism
```

**After changes**:
```bash
./scripts/compile-cuda-kernels.sh
make build-cuda
```

### GPU Clock Settings

For maximum performance, set GPU to performance mode:

```bash
# Check current power state
nvidia-smi -q -d POWER

# Set to maximum performance (requires root)
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450  # Set power limit (adjust for your GPU)
```

### Memory Pool Size

The buffer pool uses 80% of GPU memory by default. Adjust in code:

```go
// internal/gpu/cuda.go
poolMaxBytes := int64(total) * 8 / 10  // Change 8/10 to adjust percentage
```

## Usage Examples

### Basic Usage

```bash
# Use CUDA automatically (if available)
vibrant chat

# Force CUDA
vibrant chat --device cuda

# Force CPU (disable GPU)
vibrant chat --device cpu
```

### Check Device

```bash
# Show current device
vibrant device

# Test CUDA device
vibrant device --device cuda
```

### Inference

```bash
# Run inference on GPU
vibrant generate --device cuda --model qwen-2.5-coder-3b \
    --prompt "Write a function that"
```

## Testing

### Run CUDA Tests

```bash
# Compile CUDA kernels first
./scripts/compile-cuda-kernels.sh

# Run CUDA tensor tests (11 tests: device access, memory, round-trip, ops)
CGO_ENABLED=1 \
  CGO_LDFLAGS="-L$(pwd)/build/cuda -lvibrant_cuda -L/opt/cuda/lib64 -lcudart" \
  LD_LIBRARY_PATH="$(pwd)/build/cuda:/opt/cuda/lib64:$LD_LIBRARY_PATH" \
  go test -v -run "TestCUDA" ./internal/tensor/

# Run GPU device-level tests
CGO_ENABLED=1 \
  CGO_LDFLAGS="-L$(pwd)/build/cuda -lvibrant_cuda -L/opt/cuda/lib64 -lcudart" \
  LD_LIBRARY_PATH="$(pwd)/build/cuda:/opt/cuda/lib64:$LD_LIBRARY_PATH" \
  go test -v ./internal/gpu/
```

**Expected output** (RTX 4090):
```
=== RUN   TestCUDADeviceAccess
    tensor_cuda_test.go:34: CUDA device: NVIDIA GeForce RTX 4090
    tensor_cuda_test.go:40: GPU memory: 981 MB used / 24079 MB total
--- PASS: TestCUDADeviceAccess
=== RUN   TestCUDAMemoryAllocation
--- PASS: TestCUDAMemoryAllocation
=== RUN   TestCUDATensorRoundTrip
--- PASS: TestCUDATensorRoundTrip
=== RUN   TestCUDAAddGPU
--- PASS: TestCUDAAddGPU
=== RUN   TestCUDAMulGPU
--- PASS: TestCUDAMulGPU
=== RUN   TestCUDAMatMulSingleRow
--- PASS: TestCUDAMatMulSingleRow
=== RUN   TestCUDAMatMulGeneral
--- PASS: TestCUDAMatMulGeneral
=== RUN   TestCUDASoftmax
--- PASS: TestCUDASoftmax
=== RUN   TestCUDASiLU
--- PASS: TestCUDASiLU
=== RUN   TestCUDAGPUCPUConsistency
--- PASS: TestCUDAGPUCPUConsistency
=== RUN   TestCUDALargerMatMul
--- PASS: TestCUDALargerMatMul
PASS
```

## Benchmarking

### Test GPU Performance

```bash
# Build with benchmarks
CGO_ENABLED=1 \
  CGO_LDFLAGS="-L$(pwd)/build/cuda -lvibrant_cuda -L/opt/cuda/lib64 -lcudart" \
  LD_LIBRARY_PATH="$(pwd)/build/cuda:/opt/cuda/lib64:$LD_LIBRARY_PATH" \
  go test ./internal/gpu -bench=BenchmarkCUDA -benchmem -benchtime=3s

# Compare CPU vs CUDA
CGO_ENABLED=1 \
  CGO_LDFLAGS="-L$(pwd)/build/cuda -lvibrant_cuda -L/opt/cuda/lib64 -lcudart" \
  LD_LIBRARY_PATH="$(pwd)/build/cuda:/opt/cuda/lib64:$LD_LIBRARY_PATH" \
  go test ./internal/tensor -bench=BenchmarkMatMul -benchmem
```

**Expected results**:
```
BenchmarkCUDAAllocation-24           5000      236854 ns/op    8388608 B/op
BenchmarkCUDAMatMul_512x512-24       3000     3824561 ns/op          0 B/op
BenchmarkCUDAMatMul_2048x2048-24      100   187421032 ns/op          0 B/op

// CPU comparison
BenchmarkCPUMatMul_512x512-24         300    45123456 ns/op          0 B/op
// GPU is 11.8x faster!
```

## Uninstallation

### Remove CUDA Toolkit

```bash
sudo apt-get remove --purge cuda-toolkit-12-3
sudo apt-get autoremove
```

### Remove NVIDIA Drivers

```bash
sudo apt-get remove --purge nvidia-*
sudo apt-get autoremove
```

### Clean Vibrant Build

```bash
cd vibrant
make clean
rm -rf build/cuda
```

## Additional Resources

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Vibrant CUDA Backend Spec](../../specs/cuda-backend.md)
- [NVIDIA GPU Support Matrix](https://developer.nvidia.com/cuda-gpus)

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review CUDA backend spec: `specs/cuda-backend.md`
3. Check system requirements: `vibrant device --device cuda`
4. Open an issue on GitHub with:
   - `nvidia-smi` output
   - `nvcc --version` output
   - Build error messages
   - `vibrant device --device cuda` output
