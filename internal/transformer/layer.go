package transformer

import (
	"fmt"

	"github.com/xupit3r/vibrant/internal/gguf"
	"github.com/xupit3r/vibrant/internal/tensor"
)

// TransformerLayer represents a single transformer block
// Architecture: x -> norm -> attention -> residual -> norm -> ffn -> residual
type TransformerLayer struct {
	// Layer index
	layerIdx int

	// Sub-layers
	attn *Attention
	ffn  *FeedForward

	// Layer normalization
	attnNorm *RMSNorm // Pre-attention norm
	ffnNorm  *RMSNorm // Pre-FFN norm
}

// NewTransformerLayer creates a transformer layer from GGUF weights
func NewTransformerLayer(ggufFile *gguf.GGUFFile, cfg *Config, layerIdx int) (*TransformerLayer, error) {
	// Create attention layer
	attn, err := NewAttention(ggufFile, cfg, layerIdx)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention: %w", err)
	}

	// Create feed-forward layer
	ffn, err := NewFeedForward(ggufFile, cfg, layerIdx)
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN: %w", err)
	}

	// Load normalization weights (use eager dequantization for small tensors)
	prefix := fmt.Sprintf("blk.%d", layerIdx)

	attnNormWeight, err := loadTensorEager(ggufFile, prefix+".attn_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load attn norm weight: %w", err)
	}

	ffnNormWeight, err := loadTensorEager(ggufFile, prefix+".ffn_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load ffn norm weight: %w", err)
	}

	attnNorm, err := NewRMSNorm(attnNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create attn norm: %w", err)
	}

	ffnNorm, err := NewRMSNorm(ffnNormWeight, cfg.RMSNormEps)
	if err != nil {
		return nil, fmt.Errorf("failed to create ffn norm: %w", err)
	}

	return &TransformerLayer{
		layerIdx: layerIdx,
		attn:     attn,
		ffn:      ffn,
		attnNorm: attnNorm,
		ffnNorm:  ffnNorm,
	}, nil
}

// Forward computes one transformer layer
// Input: [batch_size, seq_len, hidden_dim]
// positions: [seq_len] - position indices
// useCache: whether to use KV-cache
// Output: [batch_size, seq_len, hidden_dim]
func (l *TransformerLayer) Forward(x *tensor.Tensor, positions []int, useCache bool) (*tensor.Tensor, error) {
	// Debug layer 35 specifically (enable to investigate RMSNorm convergence)
	debug := false // l.layerIdx == 35

	if debug {
		xData, _ := x.EnsureCPUData()
		xF32 := xData.([]float32)
		xMean := float32(0)
		for i := 0; i < min(100, len(xF32)); i++ {
			xMean += xF32[i]
		}
		xMean /= float32(min(100, len(xF32)))
		fmt.Printf("[L35] Input mean: %.4f\n", xMean)
	}

	// Attention block with residual connection
	// h = x + attn(norm(x))

	// Debug attn_norm weight for layer 35
	if debug {
		weightData := l.attnNorm.weight.Data().([]float32)
		wMin, wMax, wSum := weightData[0], weightData[0], float32(0)
		for _, w := range weightData[:min(100, len(weightData))] {
			if w < wMin {
				wMin = w
			}
			if w > wMax {
				wMax = w
			}
			wSum += w
		}
		wMean := wSum / float32(min(100, len(weightData)))
		fmt.Printf("[L35] Attn_norm weight: min=%.4f, max=%.4f, mean=%.4f\n", wMin, wMax, wMean)
	}

	normed, err := l.attnNorm.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("attn norm failed: %w", err)
	}

	if debug {
		nData, _ := normed.EnsureCPUData()
		nF32 := nData.([]float32)
		nMean := float32(0)
		for i := 0; i < min(100, len(nF32)); i++ {
			nMean += nF32[i]
		}
		nMean /= float32(min(100, len(nF32)))
		fmt.Printf("[L35] After attn_norm mean: %.4f\n", nMean)
	}

	attnOut, err := l.attn.Forward(normed, positions, useCache)
	if err != nil {
		return nil, fmt.Errorf("attention failed: %w", err)
	}

	if debug {
		aData, _ := attnOut.EnsureCPUData()
		aF32 := aData.([]float32)
		aMean := float32(0)
		for i := 0; i < min(100, len(aF32)); i++ {
			aMean += aF32[i]
		}
		aMean /= float32(min(100, len(aF32)))
		fmt.Printf("[L35] Attention output mean: %.4f\n", aMean)
	}

	// Residual connection
	h := addTensors(x, attnOut)

	if debug {
		hData, _ := h.EnsureCPUData()
		hF32 := hData.([]float32)
		hMean := float32(0)
		for i := 0; i < min(100, len(hF32)); i++ {
			hMean += hF32[i]
		}
		hMean /= float32(min(100, len(hF32)))
		fmt.Printf("[L35] After attn residual mean: %.4f\n", hMean)
	}

	// FFN block with residual connection
	// output = h + ffn(norm(h))
	normed, err = l.ffnNorm.Forward(h)
	if err != nil {
		return nil, fmt.Errorf("ffn norm failed: %w", err)
	}

	ffnOut, err := l.ffn.Forward(normed)
	if err != nil {
		return nil, fmt.Errorf("ffn failed: %w", err)
	}

	if debug {
		fData, _ := ffnOut.EnsureCPUData()
		fF32 := fData.([]float32)
		fMean := float32(0)
		for i := 0; i < min(100, len(fF32)); i++ {
			fMean += fF32[i]
		}
		fMean /= float32(min(100, len(fF32)))
		fmt.Printf("[L35] FFN output mean: %.4f\n", fMean)
	}

	// Residual connection
	output := addTensors(h, ffnOut)

	if debug {
		oData, _ := output.EnsureCPUData()
		oF32 := oData.([]float32)
		oMean := float32(0)
		for i := 0; i < min(100, len(oF32)); i++ {
			oMean += oF32[i]
		}
		oMean /= float32(min(100, len(oF32)))
		fmt.Printf("[L35] Final output mean: %.4f\n", oMean)
	}

	return output, nil
}

// Helper: add two tensors element-wise (residual connection)
func addTensors(a, b *tensor.Tensor) *tensor.Tensor {
	// Element-wise addition using tensor library
	return tensor.Add(a, b)
}

// ClearCache clears the KV-cache for this layer
func (l *TransformerLayer) ClearCache() {
	l.attn.ClearCache()
}

// CacheLen returns the current KV-cache length for this layer
func (l *TransformerLayer) CacheLen() int {
	return l.attn.cacheLen
}

// MoveToDevice moves layer weights to the specified device
func (l *TransformerLayer) MoveToDevice(device tensor.Device) error {
	// Move attention weights
	if err := l.attn.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move attention to device: %w", err)
	}

	// Move feedforward weights
	if err := l.ffn.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move FFN to device: %w", err)
	}

	// Move normalization weights
	if err := l.attnNorm.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move attn norm to device: %w", err)
	}

	if err := l.ffnNorm.MoveToDevice(device); err != nil {
		return fmt.Errorf("failed to move ffn norm to device: %w", err)
	}

	return nil
}
