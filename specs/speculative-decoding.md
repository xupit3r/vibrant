# Speculative Decoding & GPU/RAM Offloading Specification

## Overview

This specification describes the implementation of SpecExec (Speculative Execution) for LLM inference, along with GPU acceleration and RAM offloading capabilities for Vibrant. Based on the 2024 NeurIPS paper "SpecExec: Speculative Decoding for Interactive LLM Inference on Consumer Devices" by Svirschevski et al.

## Problem Statement

Current Vibrant limitations:
1. **Sequential token generation**: One token at a time, memory-bandwidth bound
2. **CPU-only**: No GPU acceleration for compute-heavy operations
3. **Single model**: Cannot run models larger than available RAM
4. **Slow inference**: Typical 1-5 tokens/sec on CPU

## SpecExec Algorithm

### Core Concept

SpecExec uses a small "draft" model to predict likely future tokens, then validates them in a single pass with the "target" model. Key insight: with RAM/GPU offloading, processing 1000+ tokens takes nearly the same time as 1 token.

### Algorithm Overview

```
Algorithm 1: SPECULATIVE EXECUTION

Input: prompt x, models θ_target, θ_draft, output length L, budget K, max depth D
Output: L tokens generated from θ_target distribution

1. cache := PRECOMPUTE(x, θ_draft, θ_target, K, D)
2. for t = 1 to L:
3.   if x not in cache:
4.     cache := PRECOMPUTE(x, θ_draft, θ_target, K, D)
5.   p_target := cache[x]
6.   x_next := SAMPLE(p_target)
7.   x := x ⊕ {x_next}
8. return x

function PRECOMPUTE(x, θ_target, θ_draft, K, D):
1. τ := CREATEDRAFTTREE(x, θ_draft, K, D)    # Tree with K tokens, depth D
2. next_probs := FORWARD(τ, θ_target)         # Batch forward pass
3. cache := {}
4. for each token x_i in τ:
5.   cache[prefix(x_i) ⊕ x_i] = next_probs[x_i]
6. return cache
```

### Draft Tree Construction (Algorithm 2)

Uses modified Dijkstra's algorithm to find K most probable token sequences:

```
function CREATEDRAFTTREE(x, θ_draft, K, D, B):
1. τ := TREE({x})                             # Empty tree with root x
2. H := PRIORITYQUEUE({x: 0})                 # Ordered by -log prob
3. for d = 1 to D:
4.   batch := ∅
5.   for b = 1 to B:
6.     H, x_b, nll_b := EXTRACTMIN(H)
7.     if SIZE(τ) < K:
8.       τ := ADDCHILD(τ, x_b)
9.       batch := batch ∪ {x_b}
10.  probs := FORWARD(batch, θ_draft)         # Draft model forward
11.  topk := SELECTBEST(batch, probs, K)      # Top K by cumulative prob
12.  for (x_i, p_i) in topk:
13.    H := INSERT(H, x_i, -log(p_i))
14. return TRIM(τ, K)
```

### Expected Performance

| Setup | Gen Rate | Speed | Speedup |
|-------|----------|-------|---------|
| Llama 2-7B draft / 70B target (offloaded) | 20.6 tok/step | 3.1 tok/s | 18.7x |
| Llama 2-7B / 70B GPTQ | 12.1 tok/step | 6.0 tok/s | 8.9x |
| Llama 3-8B / 70B | 18.9 tok/step | 2.6 tok/s | 15.6x |

## Architecture

### New Package Structure

```
internal/
├── speculative/           # NEW: Speculative decoding
│   ├── specexec.go        # Main SpecExec algorithm
│   ├── draft_tree.go      # Draft tree construction (Algorithm 2)
│   ├── cache.go           # Speculative cache management
│   ├── verification.go    # Token verification logic
│   └── specexec_test.go
│
├── offload/               # NEW: GPU/RAM offloading
│   ├── manager.go         # Offload orchestration
│   ├── gpu.go             # GPU memory management
│   ├── ram.go             # RAM/swap management
│   ├── scheduler.go       # Layer prefetching scheduler
│   └── offload_test.go
│
├── gpu/                   # NEW: GPU acceleration
│   ├── device.go          # Device detection & selection
│   ├── metal/             # Apple Metal backend
│   │   ├── compute.go
│   │   └── kernels.metal
│   ├── cuda/              # NVIDIA CUDA backend (future)
│   │   └── kernels.cu
│   └── gpu_test.go
│
└── inference/             # MODIFIED
    ├── engine.go          # Add speculative mode
    ├── engine_spec.go     # NEW: SpecExec engine wrapper
    └── dual_model.go      # NEW: Draft + target model management
```

### Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Engine                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Speculative Controller                  │    │
│  │  ┌─────────────┐        ┌─────────────────────┐     │    │
│  │  │ Draft Model │───────▶│   Draft Tree        │     │    │
│  │  │   (small)   │        │   Builder           │     │    │
│  │  └─────────────┘        │   (Dijkstra SSSP)   │     │    │
│  │                         └─────────┬───────────┘     │    │
│  │                                   │                 │    │
│  │                         ┌─────────▼───────────┐     │    │
│  │                         │  Speculative Cache  │     │    │
│  │                         │  (token → probs)    │     │    │
│  │                         └─────────┬───────────┘     │    │
│  │                                   │                 │    │
│  │  ┌─────────────┐        ┌─────────▼───────────┐     │    │
│  │  │Target Model │◀──────▶│    Verification     │     │    │
│  │  │  (large)    │        │    & Sampling       │     │    │
│  │  └──────┬──────┘        └─────────────────────┘     │    │
│  └─────────┼────────────────────────────────────────────┘    │
│            │                                                  │
│  ┌─────────▼──────────────────────────────────────────┐     │
│  │              Offload Manager                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │     │
│  │  │   GPU    │  │   RAM    │  │  Layer Scheduler │  │     │
│  │  │  Memory  │  │  Memory  │  │  (prefetch next) │  │     │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures

### Draft Tree

```go
// DraftTree represents a tree of speculative token continuations
type DraftTree struct {
    Root     *TreeNode
    Size     int                    // Total nodes in tree
    MaxDepth int                    // Maximum depth allowed
    Budget   int                    // Maximum nodes (K in paper)
}

// TreeNode represents a node in the draft tree
type TreeNode struct {
    Token       int                 // Token ID
    LogProb     float32             // log P(token | prefix, θ_draft)
    CumLogProb  float32             // Cumulative log probability from root
    Parent      *TreeNode
    Children    []*TreeNode
    Depth       int
}
```

### Speculative Cache

```go
// SpecCache caches target model probabilities for speculated prefixes
type SpecCache struct {
    entries map[string]*CacheEntry // prefix hash -> entry
    lru     *list.List             // LRU eviction
    maxSize int                    // Maximum cache entries
}

// CacheEntry holds cached probabilities for a prefix
type CacheEntry struct {
    PrefixHash string               // Hash of token sequence
    NextProbs  *tensor.Tensor       // [vocab_size] probability distribution
    Timestamp  time.Time            // For LRU eviction
}
```

### Offload Manager

```go
// OffloadManager handles GPU/RAM memory offloading
type OffloadManager struct {
    gpu          GPUDevice           // GPU device (Metal/CUDA)
    gpuMem       int64               // Available GPU memory
    ramMem       int64               // Available RAM for offload
    layerLocs    []DeviceLocation    // Where each layer lives
    prefetcher   *Prefetcher         // Async layer loading
}

// DeviceLocation indicates where a layer's weights are stored
type DeviceLocation int

const (
    OnGPU DeviceLocation = iota
    OnRAM
    OnDisk
)
```

## Implementation Phases

### Phase 11.1: GPU Backend Foundation
- [ ] Metal compute backend for Apple Silicon
- [ ] Device detection and capability querying
- [ ] Basic GPU tensor operations (matmul, softmax)
- [ ] Memory management (allocate, copy, free)

### Phase 11.2: RAM Offloading
- [ ] Layer-level memory management
- [ ] Async prefetching with double buffering
- [ ] Memory pressure monitoring
- [ ] Graceful degradation strategies

### Phase 11.3: Draft Tree Builder
- [ ] Priority queue implementation
- [ ] Dijkstra-based tree construction (Algorithm 2)
- [ ] Beam search variant for comparison
- [ ] Top-K token selection

### Phase 11.4: Speculative Cache
- [ ] LRU cache implementation
- [ ] Prefix hashing scheme
- [ ] Cache hit/miss statistics
- [ ] Memory-bounded eviction

### Phase 11.5: SpecExec Integration
- [ ] Dual model loading (draft + target)
- [ ] Batch forward pass for tree verification
- [ ] Token acceptance logic
- [ ] Streaming output integration

### Phase 11.6: Optimization & Tuning
- [ ] Optimal draft budget (K) selection
- [ ] Tree depth tuning (D)
- [ ] GPU/RAM partition optimization
- [ ] Profile-guided improvements

## Configuration

```yaml
# vibrant.yaml
speculative:
  enabled: true
  draft_model: "qwen2.5-coder-1.5b-q4"   # Small, fast draft model
  target_model: "qwen2.5-coder-14b-q5"    # Large, accurate target
  
  tree:
    budget: 1024        # K: max tokens in draft tree
    max_depth: 16       # D: max tree depth
    batch_size: 64      # B: draft model batch size
  
  cache:
    max_entries: 4096
    eviction: "lru"

offload:
  enabled: true
  gpu_layers: 20        # Layers to keep on GPU
  ram_budget: "32GB"    # Max RAM for offloaded weights
  prefetch: true        # Enable async prefetching
  
gpu:
  backend: "metal"      # "metal", "cuda", "vulkan", or "none"
  memory_fraction: 0.9  # Fraction of GPU memory to use
```

## API Design

### Engine Interface Extension

```go
// Engine interface extended for speculative decoding
type Engine interface {
    // Existing methods
    Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, error)
    GenerateStream(ctx context.Context, prompt string, opts GenerateOptions) (<-chan string, error)
    
    // New speculative methods
    GenerateSpeculative(ctx context.Context, prompt string, opts SpecOptions) (string, error)
    GenerateSpeculativeStream(ctx context.Context, prompt string, opts SpecOptions) (<-chan string, error)
}

// SpecOptions configures speculative generation
type SpecOptions struct {
    GenerateOptions
    TreeBudget   int    // Override default tree budget (K)
    MaxDepth     int    // Override default max depth (D)
    DraftTemp    float32 // Temperature for draft model
}
```

### Metrics & Observability

```go
// SpecMetrics tracks speculative decoding performance
type SpecMetrics struct {
    TokensGenerated    int64     // Total tokens produced
    TreesBuilt         int64     // Number of draft trees
    AvgAcceptance      float64   // Average tokens accepted per tree
    CacheHitRate       float64   // Speculative cache hit rate
    DraftTimeMs        float64   // Avg draft tree construction time
    VerifyTimeMs       float64   // Avg verification time
    OffloadTimeMs      float64   // Time spent loading weights
}
```

## Performance Targets

| Metric | Current | Target (Phase 11) |
|--------|---------|-------------------|
| Tokens/sec (CPU only) | 1-5 | 5-10 |
| Tokens/sec (GPU offload) | N/A | 10-20 |
| Tokens/sec (SpecExec) | N/A | 30-50 |
| Max model size | ~16GB | 64GB+ (with offload) |
| Memory efficiency | 100% resident | ~10-30% hot |

## Testing Strategy

### Unit Tests
- Draft tree construction correctness
- Cache operations and eviction
- Offload manager state transitions
- GPU kernel correctness

### Integration Tests
- Full SpecExec pipeline
- Multi-model coordination
- Memory pressure scenarios
- Error recovery

### Benchmarks
- Tokens/sec vs baseline
- Acceptance rate vs tree budget
- Memory bandwidth utilization
- GPU utilization

## References

1. Svirschevski et al., "SpecExec: Speculative Execution for Interactive LLM Inference on Consumer Devices", NeurIPS 2024
2. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023
3. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", 2023
4. Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference", 2023

## Open Questions

1. **Draft model selection**: What's the optimal draft/target size ratio for Vibrant's use cases?
2. **GPU backends**: Prioritize Metal (Apple) or CUDA (NVIDIA) first?
3. **Model pairs**: Should we bundle recommended draft/target pairs?
4. **Fallback**: How to handle systems without GPU gracefully?
