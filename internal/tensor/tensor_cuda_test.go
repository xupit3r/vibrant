// +build linux,cgo

package tensor

import (
	"math"
	"testing"

	"github.com/xupit3r/vibrant/internal/gpu"
)

// helper to skip test if CUDA is unavailable
func requireCUDA(t *testing.T) gpu.Device {
	t.Helper()
	dev, err := gpu.NewCUDADevice()
	if err != nil {
		t.Skipf("CUDA not available: %v", err)
	}
	return dev
}

func TestCUDADeviceAccess(t *testing.T) {
	dev := requireCUDA(t)
	defer dev.Free()

	if dev.Type() != gpu.DeviceTypeGPU {
		t.Errorf("Expected DeviceTypeGPU, got %v", dev.Type())
	}

	name := dev.Name()
	if name == "" {
		t.Error("Device name is empty")
	}
	t.Logf("CUDA device: %s", name)

	used, total := dev.MemoryUsage()
	if total == 0 {
		t.Error("Total GPU memory should be > 0")
	}
	t.Logf("GPU memory: %d MB used / %d MB total", used/(1024*1024), total/(1024*1024))
}

func TestCUDAMemoryAllocation(t *testing.T) {
	dev := requireCUDA(t)
	defer dev.Free()

	sizes := []int64{
		256,              // 256 B
		1024,             // 1 KB
		1024 * 1024,      // 1 MB
		16 * 1024 * 1024, // 16 MB
	}

	for _, size := range sizes {
		buf, err := dev.Allocate(size)
		if err != nil {
			t.Fatalf("Failed to allocate %d bytes: %v", size, err)
		}
		if buf.Size() != size {
			t.Errorf("Allocated %d bytes, but Size() reports %d", size, buf.Size())
		}
		if buf.Ptr() == 0 {
			t.Error("Buffer pointer is null")
		}
		buf.Free()
	}
}

func TestCUDATensorRoundTrip(t *testing.T) {
	requireCUDA(t)

	// Create a CPU tensor with known data
	cpuTensor := NewTensor([]int{4, 8}, Float32)
	data := cpuTensor.Data().([]float32)
	for i := range data {
		data[i] = float32(i)*0.1 + 1.0
	}

	// Move to GPU
	gpuTensor, err := cpuTensor.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move tensor to GPU: %v", err)
	}
	defer gpuTensor.FreeGPU()

	if !gpuTensor.IsOnGPU() {
		t.Fatal("Tensor should be on GPU after ToDevice(GPU)")
	}
	if gpuTensor.Device() != GPU {
		t.Errorf("Expected device GPU, got %v", gpuTensor.Device())
	}

	// Move back to CPU
	cpuResult, err := gpuTensor.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move tensor back to CPU: %v", err)
	}

	if cpuResult.Device() != CPU {
		t.Errorf("Expected device CPU, got %v", cpuResult.Device())
	}

	// Verify data integrity
	resultData := cpuResult.Data().([]float32)
	for i := range data {
		if math.Abs(float64(data[i]-resultData[i])) > 1e-6 {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, data[i], resultData[i])
			break
		}
	}
}

func TestCUDAAddGPU(t *testing.T) {
	requireCUDA(t)

	// Create test tensors
	a := NewTensor([]int{3, 4}, Float32)
	b := NewTensor([]int{3, 4}, Float32)
	aData := a.Data().([]float32)
	bData := b.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i) + 1.0
		bData[i] = float32(i) * 0.5
	}

	// Compute expected result on CPU
	expected := make([]float32, len(aData))
	for i := range expected {
		expected[i] = aData[i] + bData[i]
	}

	// Move to GPU
	gpuA, err := a.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := b.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	// Perform GPU Add
	gpuC := Add(gpuA, gpuB)
	if gpuC == nil {
		t.Fatal("GPU Add returned nil")
	}
	defer gpuC.FreeGPU()

	if !gpuC.IsOnGPU() {
		t.Error("Result should be on GPU")
	}

	// Move result back to CPU and verify
	cpuC, err := gpuC.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	cData := cpuC.Data().([]float32)
	for i := range expected {
		if math.Abs(float64(cData[i]-expected[i])) > 1e-4 {
			t.Errorf("Add mismatch at index %d: expected %f, got %f", i, expected[i], cData[i])
		}
	}
}

func TestCUDAMulGPU(t *testing.T) {
	requireCUDA(t)

	a := NewTensor([]int{3, 4}, Float32)
	b := NewTensor([]int{3, 4}, Float32)
	aData := a.Data().([]float32)
	bData := b.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i) + 1.0
		bData[i] = 2.5
	}

	expected := make([]float32, len(aData))
	for i := range expected {
		expected[i] = aData[i] * bData[i]
	}

	gpuA, err := a.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := b.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	gpuC := Mul(gpuA, gpuB)
	if gpuC == nil {
		t.Fatal("GPU Mul returned nil")
	}
	defer gpuC.FreeGPU()

	cpuC, err := gpuC.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	cData := cpuC.Data().([]float32)
	for i := range expected {
		if math.Abs(float64(cData[i]-expected[i])) > 1e-4 {
			t.Errorf("Mul mismatch at index %d: expected %f, got %f", i, expected[i], cData[i])
		}
	}
}

func TestCUDAMatMulSingleRow(t *testing.T) {
	requireCUDA(t)

	// A: [1 x 4], B: [4 x 3] -> C: [1 x 3]
	A := NewTensor([]int{1, 4}, Float32)
	aData := A.Data().([]float32)
	aData[0], aData[1], aData[2], aData[3] = 1, 2, 3, 4

	B := NewTensor([]int{4, 3}, Float32)
	bData := B.Data().([]float32)
	for i := range bData {
		bData[i] = float32(i + 1)
	}
	// B = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	// C[0] = 1*1 + 2*4 + 3*7 + 4*10 = 70
	// C[1] = 1*2 + 2*5 + 3*8 + 4*11 = 80
	// C[2] = 1*3 + 2*6 + 3*9 + 4*12 = 90
	expected := []float32{70, 80, 90}

	gpuA, err := A.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := B.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	gpuC := MatMul(gpuA, gpuB)
	if gpuC == nil {
		t.Fatal("GPU MatMul returned nil")
	}
	defer gpuC.FreeGPU()

	cpuC, err := gpuC.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	cData := cpuC.Data().([]float32)
	for i := range expected {
		if math.Abs(float64(cData[i]-expected[i])) > 1e-4 {
			t.Errorf("MatMul mismatch at index %d: expected %f, got %f", i, expected[i], cData[i])
		}
	}
}

func TestCUDAMatMulGeneral(t *testing.T) {
	requireCUDA(t)

	// A: [2 x 3], B: [3 x 2] -> C: [2 x 2]
	A := NewTensor([]int{2, 3}, Float32)
	aData := A.Data().([]float32)
	aData[0], aData[1], aData[2] = 1, 2, 3
	aData[3], aData[4], aData[5] = 4, 5, 6

	B := NewTensor([]int{3, 2}, Float32)
	bData := B.Data().([]float32)
	bData[0], bData[1] = 1, 2
	bData[2], bData[3] = 3, 4
	bData[4], bData[5] = 5, 6

	// C[0,0] = 1*1 + 2*3 + 3*5 = 22
	// C[0,1] = 1*2 + 2*4 + 3*6 = 28
	// C[1,0] = 4*1 + 5*3 + 6*5 = 49
	// C[1,1] = 4*2 + 5*4 + 6*6 = 64
	expected := []float32{22, 28, 49, 64}

	gpuA, err := A.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := B.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	gpuC := MatMul(gpuA, gpuB)
	if gpuC == nil {
		t.Fatal("GPU MatMul returned nil")
	}
	defer gpuC.FreeGPU()

	cpuC, err := gpuC.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	cData := cpuC.Data().([]float32)
	for i := range expected {
		if math.Abs(float64(cData[i]-expected[i])) > 1e-2 {
			t.Errorf("MatMul mismatch at index %d: expected %f, got %f", i, expected[i], cData[i])
		}
	}
}

func TestCUDASoftmax(t *testing.T) {
	requireCUDA(t)

	input := NewTensor([]int{5}, Float32)
	data := input.Data().([]float32)
	data[0], data[1], data[2], data[3], data[4] = 1.0, 2.0, 3.0, 4.0, 5.0

	// Compute expected softmax on CPU
	cpuResult := Softmax(input)
	expectedData := cpuResult.Data().([]float32)

	// Move to GPU and compute
	gpuInput, err := input.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move input to GPU: %v", err)
	}
	defer gpuInput.FreeGPU()

	gpuResult := softmaxGPU(gpuInput)
	if gpuResult == nil {
		t.Fatal("GPU Softmax returned nil")
	}
	defer gpuResult.FreeGPU()

	cpuOut, err := gpuResult.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	resultData := cpuOut.Data().([]float32)

	// Verify softmax properties: all positive, sums to 1
	var sum float32
	for i, val := range resultData {
		if val < 0 || val > 1 {
			t.Errorf("Softmax output[%d] = %f, expected value in [0,1]", i, val)
		}
		sum += val
	}
	if math.Abs(float64(sum-1.0)) > 1e-3 {
		t.Errorf("Softmax sum = %f, expected 1.0", sum)
	}

	// Verify against CPU reference
	for i := range expectedData {
		if math.Abs(float64(resultData[i]-expectedData[i])) > 1e-4 {
			t.Errorf("Softmax mismatch at index %d: expected %f, got %f", i, expectedData[i], resultData[i])
		}
	}
}

func TestCUDASiLU(t *testing.T) {
	requireCUDA(t)

	input := NewTensor([]int{6}, Float32)
	data := input.Data().([]float32)
	data[0], data[1], data[2] = -2.0, -1.0, 0.0
	data[3], data[4], data[5] = 1.0, 2.0, 3.0

	// Compute expected SiLU on CPU: x * sigmoid(x)
	cpuResult := SiLU(input)
	expectedData := cpuResult.Data().([]float32)

	// Move to GPU and compute
	gpuInput, err := input.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move input to GPU: %v", err)
	}
	defer gpuInput.FreeGPU()

	gpuResult := siluGPU(gpuInput)
	if gpuResult == nil {
		t.Fatal("GPU SiLU returned nil")
	}
	defer gpuResult.FreeGPU()

	cpuOut, err := gpuResult.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	resultData := cpuOut.Data().([]float32)
	for i := range expectedData {
		if math.Abs(float64(resultData[i]-expectedData[i])) > 1e-4 {
			t.Errorf("SiLU mismatch at index %d: expected %f, got %f", i, expectedData[i], resultData[i])
		}
	}
}

func TestCUDAGPUCPUConsistency(t *testing.T) {
	requireCUDA(t)

	// Run the same operations on CPU and GPU, compare results

	// Create input tensors
	a := NewTensor([]int{8, 16}, Float32)
	b := NewTensor([]int{8, 16}, Float32)
	aData := a.Data().([]float32)
	bData := b.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i%7)*0.3 - 1.0
		bData[i] = float32(i%5)*0.2 + 0.5
	}

	// CPU results
	cpuAdd := Add(a, b)
	cpuMul := Mul(a, b)

	// GPU results
	gpuA, err := a.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := b.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	gpuAddResult := Add(gpuA, gpuB)
	if gpuAddResult == nil {
		t.Fatal("GPU Add returned nil")
	}
	defer gpuAddResult.FreeGPU()

	gpuMulResult := Mul(gpuA, gpuB)
	if gpuMulResult == nil {
		t.Fatal("GPU Mul returned nil")
	}
	defer gpuMulResult.FreeGPU()

	// Transfer GPU results to CPU
	addBack, err := gpuAddResult.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move Add result to CPU: %v", err)
	}
	mulBack, err := gpuMulResult.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move Mul result to CPU: %v", err)
	}

	// Compare
	cpuAddData := cpuAdd.Data().([]float32)
	gpuAddData := addBack.Data().([]float32)
	for i := range cpuAddData {
		if math.Abs(float64(cpuAddData[i]-gpuAddData[i])) > 1e-4 {
			t.Errorf("Add CPU/GPU mismatch at index %d: CPU=%f, GPU=%f", i, cpuAddData[i], gpuAddData[i])
			break
		}
	}

	cpuMulData := cpuMul.Data().([]float32)
	gpuMulData := mulBack.Data().([]float32)
	for i := range cpuMulData {
		if math.Abs(float64(cpuMulData[i]-gpuMulData[i])) > 1e-4 {
			t.Errorf("Mul CPU/GPU mismatch at index %d: CPU=%f, GPU=%f", i, cpuMulData[i], gpuMulData[i])
			break
		}
	}
}

func TestCUDALargerMatMul(t *testing.T) {
	requireCUDA(t)

	// Test with a moderately sized matrix to exercise the tiled kernel
	M, K, N := 32, 64, 32
	A := NewTensor([]int{M, K}, Float32)
	B := NewTensor([]int{K, N}, Float32)

	aData := A.Data().([]float32)
	bData := B.Data().([]float32)
	for i := range aData {
		aData[i] = float32(i%10) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i%10) * 0.1
	}

	// CPU reference
	cpuC := MatMul(A, B)
	cpuCData := cpuC.Data().([]float32)

	// GPU computation
	gpuA, err := A.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move A to GPU: %v", err)
	}
	defer gpuA.FreeGPU()

	gpuB, err := B.ToDevice(GPU)
	if err != nil {
		t.Fatalf("Failed to move B to GPU: %v", err)
	}
	defer gpuB.FreeGPU()

	gpuC := MatMul(gpuA, gpuB)
	if gpuC == nil {
		t.Fatal("GPU MatMul returned nil")
	}
	defer gpuC.FreeGPU()

	gpuCBack, err := gpuC.ToDevice(CPU)
	if err != nil {
		t.Fatalf("Failed to move result to CPU: %v", err)
	}

	gpuCData := gpuCBack.Data().([]float32)
	mismatches := 0
	for i := range cpuCData {
		diff := math.Abs(float64(cpuCData[i] - gpuCData[i]))
		// Use a relative tolerance for larger values
		relTol := math.Max(1e-3, 1e-3*math.Abs(float64(cpuCData[i])))
		if diff > relTol {
			if mismatches < 5 {
				t.Errorf("MatMul mismatch at [%d,%d]: CPU=%f, GPU=%f (diff=%f)",
					i/N, i%N, cpuCData[i], gpuCData[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("Total MatMul mismatches: %d / %d", mismatches, len(cpuCData))
	}
}
