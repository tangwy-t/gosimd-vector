# gosimd-vector Design Spec

> 基于 Go 1.26 SIMD 的高性能嵌入式向量检索引擎

**Date:** 2026-06-08
**Status:** Draft
**Go Version:** 1.26+
**Dependencies:** Go standard library (`simd/archsimd`, `math/rand/v2`) + `golang.org/x/sys/cpu`

---

## Overview

gosimd-vector 是一个纯 Go 实现的嵌入式向量检索引擎，利用 Go 1.26 标准库中的 `simd/archsimd` 包提供 CPU SIMD 加速的向量距离计算，并在此基础上构建 HNSW 索引以支持百万级向量的近似最近邻（ANN）搜索。

**核心定位：** 不做向量数据库（已有 Milvus、Qdrant），而是做一个嵌入式的高性能向量计算引擎，可以被其他 Go 项目直接 import 使用。

**关键特性：**
- 零 CGO 依赖，纯 Go 实现
- 运行时 CPU 能力检测 + 函数指针派发（单一二进制，自动适配 AVX2/AVX-512/NEON）
- 理论加速 4-8x（相比纯标量实现）
- 函数式 API，简单直接

---

## Architecture

### 分层架构

```
search.Engine  (公共 API 层)
    |
    v
index.HNSWIndex  (索引层：HNSW 图构建与搜索)
    |
    v
distance.DotProduct/L2/Cosine  (SIMD 加速层)
    |
    v
simd/archsimd  (Go 1.26 标准库)
```

### 包结构

```
gosimd-vector/
├── go.mod                        # module github.com/<your-github>/gosimd-vector (更新为实际 GitHub 用户名)
├── distance/                     # Phase 1: SIMD 距离计算内核
│   ├── distance.go               # 公共 API
│   ├── dispatch.go               # init() CPU 检测 + 函数指针表
│   ├── distance_avx2.go          # Float32x8 实现 (AVX2+FMA)
│   ├── distance_avx512.go        # Float32x16 实现 (AVX-512)
│   ├── distance_neon.go          # Float32x4 实现 (ARM64 NEON)
│   ├── distance_scalar.go        # 纯标量 fallback
│   └── distance_test.go          # 单元测试 + benchmark
├── index/                        # Phase 2: HNSW 索引
│   ├── hnsw.go                   # HNSW 图结构与构建
│   ├── hnsw_search.go            # 图搜索（贪心 + beam search）
│   ├── graph.go                  # 多层图数据结构 + 连续向量存储
│   └── hnsw_test.go
├── search/                       # Phase 2: 搜索 API
│   ├── engine.go                 # SearchEngine 封装
│   └── engine_test.go
└── benchmark/                    # Benchmark 套件
    ├── bench_test.go             # Go benchmark
    └── bench_compare.go          # scalar vs SIMD 对比报告生成
```

---

## Phase 1: SIMD Distance Computation Kernel

### Public API

```go
package distance

// MetricType 标识距离度量类型
type MetricType int
const (
    L2 MetricType = iota
    Cosine
    InnerProduct
)

// DotProduct 返回两个向量的内积
func DotProduct(a, b []float32) float32

// CosineSimilarity 返回余弦相似度，范围 [-1, 1]
func CosineSimilarity(a, b []float32) float32

// L2Distance 返回欧几里得距离
func L2Distance(a, b []float32) float32

// L2Squared 返回 L2 距离的平方（跳过 sqrt，用于比较场景更高效）
func L2Squared(a, b []float32) float32

// Normalize 预归一化向量（原地修改）
// 归一化后 DotProduct 等价于 CosineSimilarity
func Normalize(v []float32)
```

### Runtime Dispatch Mechanism

```go
// dispatch.go
import "golang.org/x/sys/cpu"

type dotProductFn func(a, b []float32) float32
var dotProductImpl dotProductFn

func init() {
    if cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ {
        dotProductImpl = dotProductAVX512
    } else if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
        dotProductImpl = dotProductAVX2
    } else {
        dotProductImpl = dotProductScalar
    }
}

func DotProduct(a, b []float32) float32 { return dotProductImpl(a, b) }
```

派发策略选择运行时函数指针的理由：
1. 对库消费者最友好，`go get` 即用，无需特殊编译参数
2. HNSW 搜索中距离计算在热循环外被调用，函数指针 1-2ns 的间接开销相对于完整的向量计算（~50-200ns for dim=768）可忽略
3. 便于 benchmark 时切换对比不同实现

### SIMD Implementations

#### AVX2 (Float32x8, 8-way parallelism)

```go
// distance_avx2.go
//go:build amd64

func dotProductAVX2(a, b []float32) float32 {
    n := len(a)
    sum := archsimd.BroadcastFloat32x8(0)
    
    i := 0
    for ; i+8 <= n; i += 8 {
        va := archsimd.LoadFloat32x8Slice(a[i:])
        vb := archsimd.LoadFloat32x8Slice(b[i:])
        sum = sum.MulAdd(va, vb)
    }
    
    lo := sum.GetLo()
    hi := sum.GetHi()
    v4 := lo.Add(hi)
    v2 := v4.AddPairs(v4)
    v1 := v2.AddPairs(v2)
    result := v1.GetElem(0)
    
    for ; i < n; i++ {
        result += a[i] * b[i]
    }
    return result
}
```

#### AVX-512 (Float32x16, 16-way parallelism)

```go
// distance_avx512.go
//go:build amd64

func dotProductAVX512(a, b []float32) float32 {
    n := len(a)
    sum := archsimd.BroadcastFloat32x16(0)
    
    i := 0
    for ; i+16 <= n; i += 16 {
        va := archsimd.LoadFloat32x16Slice(a[i:])
        vb := archsimd.LoadFloat32x16Slice(b[i:])
        sum = sum.MulAdd(va, vb)
    }
    
    lo := sum.GetLo()
    hi := sum.GetHi()
    sum8 := lo.Add(hi)
    l8 := sum8.GetLo()
    h8 := sum8.GetHi()
    v4 := l8.Add(h8)
    v2 := v4.AddPairs(v4)
    v1 := v2.AddPairs(v2)
    result := v1.GetElem(0)
    
    for ; i < n; i++ {
        result += a[i] * b[i]
    }
    return result
}
```

#### ARM64 NEON (Float32x4, 4-way parallelism)

```go
// distance_neon.go
//go:build arm64

func dotProductNEON(a, b []float32) float32 {
    n := len(a)
    sum := archsimd.BroadcastFloat32x4(0)
    
    i := 0
    for ; i+4 <= n; i += 4 {
        va := archsimd.LoadFloat32x4Slice(a[i:])
        vb := archsimd.LoadFloat32x4Slice(b[i:])
        sum = sum.MulAdd(va, vb)
    }
    
    v2 := sum.AddPairs(sum)
    v1 := v2.AddPairs(v2)
    result := v1.GetElem(0)
    
    for ; i < n; i++ {
        result += a[i] * b[i]
    }
    return result
}
```

#### Scalar Fallback

```go
// distance_scalar.go
func dotProductScalar(a, b []float32) float32 {
    var sum float32
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}
```

### Cosine Similarity 与 L2 Distance

CosineSimilarity 使用单次遍历同时计算 `dot(a,b)`、`dot(a,a)`、`dot(b,b)` 以提升效率：

```go
func cosineSimilaritySIMD(a, b []float32) float32 {
    dot, normA, normB := dotAndNormsImpl(a, b)
    denom := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
    if denom == 0 {
        return 0
    }
    return dot / denom
}
```

L2Squared 采用直接计算方式（`Sub` + `MulAdd`），与 DotProduct 结构一致，便于代码复用和水平求和逻辑共享：

```go
func l2SquaredAVX2(a, b []float32) float32 {
    n := len(a)
    sum := archsimd.BroadcastFloat32x8(0)
    i := 0
    for ; i+8 <= n; i += 8 {
        va := archsimd.LoadFloat32x8Slice(a[i:])
        vb := archsimd.LoadFloat32x8Slice(b[i:])
        diff := va.Sub(vb)
        sum = sum.MulAdd(diff, diff) // sum += (a-b)^2
    }
    // horizontal sum (same as DotProduct) + tail with scalar loop
}
```

L2Distance 在此基础上调用 `math.Sqrt`：

```go
func L2Distance(a, b []float32) float32 {
    return float32(math.Sqrt(float64(l2SquaredImpl(a, b))))
}
```

### Tail Handling Strategy

- **主循环**：每次处理 SIMD 宽度整数倍的元素（8 for AVX2, 16 for AVX-512）
- **尾部元素（`len % width`）**：使用标量循环处理，避免 masked load 的潜在额外开销
- 如 benchmark 表明 masked load 更优（dim 较小时），改用 `LoadFloat32x8SlicePart`

---

## Phase 2: HNSW Index

### Graph Data Structure

```go
// graph.go
type layer struct {
    neighbors [][]int32
}

type graph struct {
    layers     []layer
    maxLayer   int
    store      *vectorStore
    entryPoint int32
    dim        int
    metric     distance.MetricType
}

type vectorStore struct {
    data  []float32
    dim   int
    count int
}
```

向量存储采用单一连续 `[]float32` 切片，按维度对齐。这确保：
1. SIMD `LoadFloat32x8Slice` 可直接加载，无额外拷贝
2. 缓存友好：相邻节点在内存中相邻
3. 单一内存分配，减少 GC 压力

### HNSW Parameters

| 参数 | 默认值 | 含义 |
|------|--------|------|
| M | 16 | 每个节点的最大连接数 |
| efConstruction | 200 | 构建时的 beam 宽度 |
| efSearch | 64 | 搜索时的 beam 宽度 |
| levelMult | 1/ln(M) | 层数分配系数 |

高层（非底层）M 减半为 M/2（遵循 HNSW 原始论文）。

### Construction Algorithm

```
function insert(vector, id):
    level = floor(-ln(uniform(0,1)) * levelMult)
    
    with writeLock:
        curr = entryPoint
        // 从顶层到 level+1 层：贪心搜索
        for l from maxLayer down to level+1:
            curr = greedySearch(curr, vector, l)
        
        // 从 level 到 0 层：beam search + 添加连接
        for l from min(level, maxLayer) down to 0:
            neighbors = beamSearch(curr, vector, l, efConstruction)
            addConnections(id, neighbors, l, M)
            curr = neighbors[0]
        
        update entryPoint if level > maxLayer
```

### Search Algorithm

```
function search(query, k, efSearch):
    with readLock:
        curr = entryPoint
        // 从顶层到第1层：贪心搜索
        for l from maxLayer down to 1:
            curr = greedySearch(curr, query, l)
        
        // 底层 beam search
        candidates = beamSearch(curr, query, 0, efSearch)
        return topK(candidates, k)
```

### Concurrency

- 搜索使用 `sync.RWMutex` 的 `RLock`，支持并发查询
- 插入/删除使用 `Lock`，串行化修改

### Serialization

二进制格式（little-endian）：

```
[Magic(4)] [Version(4)] [Dim(4)] [Metric(1)] [M(4)] [efC(4)]
[NodeCount(4)] [EntryPoint(4)]
[Layers(4)]
[for each layer: [for each node: [neighborCount(4)] [neighbors...]]]
[VectorData: dim * nodeCount * 4 bytes]
```

```go
func (idx *HNSWIndex) Save(w io.Writer) error
func LoadIndex(r io.Reader) (*HNSWIndex, error)
```

---

## Phase 2: Search Engine API

```go
package search

type SearchResult struct {
    ID       int32
    Distance float32
}

type Engine struct { ... }

func New(dim int, metric distance.MetricType, opts ...Option) *Engine

type Option func(*config)
func WithM(m int) Option
func WithEfConstruction(ef int) Option
func WithEfSearch(ef int) Option

func (e *Engine) Add(id int32, vector []float32) error
func (e *Engine) AddBatch(vectors map[int32][]float32) error
func (e *Engine) Search(query []float32, k int) []SearchResult
func (e *Engine) SearchWithEf(query []float32, k int, efSearch int) []SearchResult
func (e *Engine) Remove(id int32) error
func (e *Engine) Count() int
func (e *Engine) Save(path string) error
func (e *Engine) Load(path string) error
```

### Usage Example

```go
import "github.com/<your-github>/gosimd-vector/search"

engine := search.New(768, distance.Cosine,
    search.WithM(16),
    search.WithEfConstruction(200),
)

// 批量添加
engine.AddBatch(vectors)

// 搜索 top-10
results := engine.Search(query, 10)
for _, r := range results {
    fmt.Printf("ID=%d Distance=%.4f\n", r.ID, r.Distance)
}

// 持久化
engine.Save("my_index.bin")
```

---

## Benchmark Plan

```go
// 1. 距离计算内核对比（核心 benchmark）
func BenchmarkDotProduct(b *testing.B) {
    for _, dim := range []int{128, 256, 384, 768, 1536} {
        for _, impl := range []string{"Scalar", "AVX2", "AVX512"} {
            b.Run(fmt.Sprintf("dim%d_%s", dim, impl), ...)
        }
    }
}

// 2. 完整距离度量对比
func BenchmarkDistance(b *testing.B) {
    for _, metric := range []string{"DotProduct", "Cosine", "L2"} {
        for _, impl := range []string{"Scalar", "SIMD"} { ... }
    }
}

// 3. HNSW 端到端搜索性能
func BenchmarkSearch(b *testing.B) {
    for _, count := range []int{10_000, 100_000, 1_000_000} {
        b.Run(fmt.Sprintf("n%d", count), ...)
    }
}

// 4. Recall@K vs QPS 曲线（评估索引质量）
func BenchmarkRecallVsQPS(b *testing.B) {
    for _, ef := range []int{16, 32, 64, 128, 256} {
        b.Run(fmt.Sprintf("ef%d", ef), ...)
    }
}

// 5. 并发搜索性能
func BenchmarkSearch_Concurrent(b *testing.B) {
    for _, goroutines := range []int{1, 4, 8, 16, 32} {
        b.Run(fmt.Sprintf("g%d", goroutines), ...)
    }
}
```

目标性能指标：
- DotProduct(dim=768, 100万向量)：SIMD vs Scalar 加速 4-8x
- HNSW 搜索(100万向量, dim=768, k=10)：<1ms/query @recall>0.95

---

## Phased Delivery Plan

### Phase 1 (MVP): SIMD 距离内核 + Benchmark
- `distance/` 包完整实现（所有 metric × 所有架构）
- 完整的单元测试（正确性验证，包括 NaN/Inf edge cases）
- Benchmark 对比报告

### Phase 2: HNSW 索引 + 搜索
- `index/` HNSW 图实现（构建 + 搜索 + 序列化）
- `search/` Engine API 封装
- 端到端 benchmark + recall 评估

---

## Dependencies

| 依赖 | 版本 | 用途 |
|------|------|------|
| `simd/archsimd` | Go 1.26 stdlib | SIMD 指令 |
| `golang.org/x/sys/cpu` | - | CPU 能力运行时检测（AVX2/AVX-512/FMA flags） |
| `math/rand/v2` | Go 1.26 stdlib | HNSW 层数随机分配 |
| 无 CGO 依赖 | - | 纯 Go |

---

## Risks and Mitigations

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| `archsimd` 仍为实验性 API | 未来 Go 版本可能变化 | Phase 1 先发布，用 build tag 隔离 SIMD 依赖，fallback 到标量 |
| AVX-512 在部分 CPU 降频 | 实际加速不如预期 | 运行时性能检测可关闭 AVX-512，回退到 AVX-2 |
| HNSW 构建内存压力大 | 百万向量 M=16 → 约 ~300MB 邻居存储 | 考虑 mmap 或分块加载（Phase 2 优化） |
| Go 编译器 SIMD 代码生成未优化 | SIMD 加速不如理论值 | 关键路径用 `_test.go` 验证汇编输出 |
