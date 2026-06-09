package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gosimd-vector/gosimd-vector/distance"
	"github.com/gosimd-vector/gosimd-vector/search"
)

func main() {
	fmt.Println("=== gosimd-vector Demo ===")
	fmt.Println("基于 Go 1.26 SIMD 的高性能向量检索引擎")
	fmt.Println()

	// 1. 距离计算演示
	demonstrateDistances()

	// 2. 性能对比演示
	demonstratePerformance()

	// 3. HNSW 搜索引擎演示
	demonstrateSearch()
}

func demonstrateDistances() {
	fmt.Println("1️⃣  向量距离计算")
	fmt.Println("────────────────────────────────────")

	// 两个示例向量
	a := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	b := []float32{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}

	fmt.Printf("向量 A: %v\n", a[:4])
	fmt.Printf("向量 B: %v\n", b[:4])
	fmt.Printf("维度: %d\n", len(a))
	fmt.Println()

	// 计算三种距离
	dot := distance.DotProduct(a, b)
	cosine := distance.CosineSimilarity(a, b)
	l2 := distance.L2Distance(a, b)
	l2sq := distance.L2Squared(a, b)

	fmt.Printf("• Dot Product (内积):     %.4f\n", dot)
	fmt.Printf("• Cosine Similarity:      %.4f\n", cosine)
	fmt.Printf("• L2 Distance (欧氏距离):  %.4f\n", l2)
	fmt.Printf("• L2 Squared (距离平方):   %.4f\n", l2sq)
	fmt.Println()
}

func demonstratePerformance() {
	fmt.Println("2️⃣  SIMD 加速性能对比")
	fmt.Println("────────────────────────────────────")

	dim := 768
	a := randomVector(dim, 42)
	b := randomVector(dim, 99)

	fmt.Printf("测试维度: %d\n", dim)
	fmt.Println()

	// 预热
	for i := 0; i < 1000; i++ {
		_ = distance.DotProduct(a, b)
		_ = distance.CosineSimilarity(a, b)
		_ = distance.L2Distance(a, b)
	}

	// DotProduct
	iterations := 100000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = distance.DotProduct(a, b)
	}
	dotDuration := time.Since(start)

	// Cosine
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = distance.CosineSimilarity(a, b)
	}
	cosineDuration := time.Since(start)

	// L2
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_ = distance.L2Distance(a, b)
	}
	l2Duration := time.Since(start)

	fmt.Printf("执行 %d 次计算:\n", iterations)
	fmt.Printf("• DotProduct:     %10v (平均 %v/次)\n", dotDuration, dotDuration/time.Duration(iterations))
	fmt.Printf("• CosineSimilarity:%9v (平均 %v/次) [融合优化]\n", cosineDuration, cosineDuration/time.Duration(iterations))
	fmt.Printf("• L2Distance:     %10v (平均 %v/次)\n", l2Duration, l2Duration/time.Duration(iterations))
	fmt.Println()
}

func demonstrateSearch() {
	fmt.Println("3️⃣  HNSW 近似最近邻搜索")
	fmt.Println("────────────────────────────────────")

	dim := 128
	numVectors := 10000
	k := 5

	fmt.Printf("向量库大小: %d\n", numVectors)
	fmt.Printf("维度: %d\n", dim)
	fmt.Printf("搜索 K: %d\n\n", k)

	// 创建搜索引擎
	engine := search.New(
		dim,
		distance.Cosine,
		search.WithM(16),
		search.WithEfConstruction(200),
		search.WithEfSearch(50),
	)

	// 批量添加向量
	fmt.Print("添加向量...")
	start := time.Now()
	vectors := makeVectors(numVectors, dim, 12345)
	for id, vec := range vectors {
		engine.Add(int32(id), vec)
	}
	addDuration := time.Since(start)
	fmt.Printf(" ✓ (%v)\n", addDuration)

	// 执行搜索
	fmt.Print("搜索最近邻...")
	query := randomVector(dim, 99999)

	searchStart := time.Now()
	results := engine.Search(query, k)
	searchDuration := time.Since(searchStart)
	fmt.Printf(" ✓ (%v/次)\n", searchDuration)

	fmt.Println("\n搜索结果:")
	for i, result := range results {
		fmt.Printf("  #%d - 向量ID: %4d, 距离: %.6f\n", i+1, result.ID, result.Distance)
	}

	// 并发搜索测试
	fmt.Println("\n并发搜索测试 (8 goroutines)...")
	concurrentStart := time.Now()
	iterations := 10000
	done := make(chan bool, 8)

	for g := 0; g < 8; g++ {
		go func() {
			for i := 0; i < iterations/8; i++ {
				_ = engine.Search(query, k)
			}
			done <- true
		}()
	}

	for i := 0; i < 8; i++ {
		<-done
	}
	concurrentDuration := time.Since(concurrentStart)

	qps := float64(iterations) / concurrentDuration.Seconds()
	fmt.Printf("  QPS: %.0f queries/sec\n", qps)
	fmt.Printf("  平均延迟: %v/query\n", concurrentDuration/time.Duration(iterations))

	fmt.Println("\n✅ Demo 完成!")
}

// ===== 辅助函数 =====

func randomVector(dim int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = r.Float32()
	}
	return vec
}

func makeVectors(count, dim int, seed int64) map[int32][]float32 {
	r := rand.New(rand.NewSource(seed))
	vectors := make(map[int32][]float32, count)

	for id := int32(0); id < int32(count); id++ {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = r.Float32()
		}
		vectors[id] = vec
	}

	return vectors
}
