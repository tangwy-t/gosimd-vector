# gosimd-vector

[![Go Reference](https://pkg.go.dev/badge/github.com/gosimd-vector/gosimd-vector.svg)](https://pkg.go.dev/github.com/gosimd-vector/gosimd-vector)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

基于 Go 1.26 `simd/archsimd` 的高性能向量检索引擎。零 CGO，纯 Go 实现。

> ⚠️ **实验性项目** — 依赖 `GOEXPERIMENT=simd`，需要 Go 1.26+

## 性能（AMD Ryzen 7 7840HS，dim=768，零堆分配）

| 函数 | 标量 | 融合 AVX-512 | 加速比 |
|:-----|:----:|:------------:|:------:|
| **DotProduct** | 207 ns | 72 ns | **2.9x** |
| **CosineSimilarity** | 1719 ns | 91 ns | **19x** |
| **HNSW Search** (vs 暴力搜索) | 1.21 ms | 107 μs | **11.3x** |

## 快速开始

```bash
# 需要 GOEXPERIMENT=simd
export GOEXPERIMENT=simd
```

```go
import (
    "github.com/gosimd-vector/gosimd-vector/search"
    "github.com/gosimd-vector/gosimd-vector/distance"
)

// 创建搜索引擎
engine := search.New(768, distance.Cosine,
    search.WithM(16),
    search.WithEfConstruction(200),
)

// 添加向量
engine.Add(0, myVector768)

// 搜索 top-10
results := engine.Search(queryVector, 10)
```

## 包结构

```
gosimd-vector/
├── distance/    SIMD 距离计算内核（DotProduct, CosineSimilarity, L2Distance）
├── index/       HNSW 索引（Add, Remove, Search, Save/Load）
└── search/      高层 API（Engine，自动余弦归一化）
```

- **distance** — 直接使用 SIMD 加速的距离函数
- **index** — HNSW 图索引，支持序列化
- **search** — 用户友好的 Engine 封装

## 距离度量

| 度量 | 函数 | 公式 |
|:-----|:-----|:-----|
| L2 | `L2Distance(a, b)` | `√(Σ(aᵢ - bᵢ)²)` |
| Cosine | `CosineSimilarity(a, b)` | `(a·b) / (‖a‖·‖b‖)` |
| Inner Product | `DotProduct(a, b)` | `Σ(aᵢ·bᵢ)` |

## HNSW 参数

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| M | 16 | 每节点最大连接数 |
| EfConstruction | 200 | 构建 beam 宽度（越高越慢，质量越好） |
| EfSearch | 50 | 搜索 beam 宽度（越高越慢，召回越好） |

## 序列化

```go
// 保存
engine.Save("my_index.bin")

// 加载
engine.Load("my_index.bin")
```

## 持久化格式

二进制 little-endian，包含 magic number、版本号、图结构、向量数据。

## 构建与测试

```bash
export GOEXPERIMENT=simd

go test ./...                    # 运行所有测试
go test ./distance/ -v           # 距离计算测试
go test ./index/ -v              # HNSW 测试
go test ./search/ -v             # 引擎测试

# Benchmark
go test ./distance/ -bench=. -benchmem
go test ./index/ -bench=. -benchmem
```

## 架构支持

| 架构 | SIMD 级别 | 状态 |
|:-----|:----------|:----:|
| amd64 (AVX-512) | Float32x16 (512-bit) | ✅ 优化 |
| amd64 (AVX2+FMA) | Float32x8 (256-bit) | ✅ 优化 |
| arm64 | 标量 fallback | ⏳ 等待 Go archsimd 支持 |

## 已知限制

- 需要 `GOEXPERIMENT=simd`（实验性 API）
- ARM64 SIMD 暂不可用（Go 标准库尚未支持）
- 单线程 Add（内部 `sync.Mutex`），搜索可并发（`sync.RWMutex`）
- 不支持动态维度变更（创建时固定）

## 许可证

[MIT](LICENSE)
