package index

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

func BenchmarkHNSW_Add(b *testing.B) {
	dim := 128
	for _, n := range []int{1_000, 10_000, 50_000} {
		b.Run(fmt.Sprintf("n=%d_dim=%d", n, dim), func(b *testing.B) {
			vecs := make([][]float32, n)
			for i := range vecs {
				vecs[i] = makeRandomVec(dim, int64(i))
			}
			b.ResetTimer()
			for range b.N {
				h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
				for i, v := range vecs {
					h.Add(int32(i), v)
				}
			}
		})
	}
}

func BenchmarkHNSW_Search_Dim128(b *testing.B) {
	dim := 128
	for _, n := range []int{10_000, 100_000} {
		h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
		for i := 0; i < n; i++ {
			h.Add(int32(i), makeRandomVec(dim, int64(i)))
		}
		query := makeRandomVec(dim, 99999)
		b.Run(fmt.Sprintf("n=%d_k=10", n), func(b *testing.B) {
			for range b.N {
				h.Search(query, 10)
			}
		})
	}
}

func BenchmarkHNSW_Search_EfSweep(b *testing.B) {
	dim := 128
	n := 10_000
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
	for i := 0; i < n; i++ {
		h.Add(int32(i), makeRandomVec(dim, int64(i)))
	}
	query := makeRandomVec(dim, 99999)
	for _, ef := range []int{10, 20, 50, 100, 200} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			for range b.N {
				h.SearchWithEf(query, 10, ef)
			}
		})
	}
}

func BenchmarkHNSW_RecallAtEf(b *testing.B) {
	dim := 128
	n := 10_000
	k := 10
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
	store := newVectorStore(dim, n)
	for i := 0; i < n; i++ {
		v := makeRandomVec(dim, int64(i))
		h.Add(int32(i), v)
		store.add(v)
	}
	distFn := selectDistFn(distance.L2)

	numQ := 50
	queries := make([][]float32, numQ)
	for i := range queries {
		queries[i] = makeRandomVec(dim, int64(1_000_000+i))
	}

	bruteResults := make([][]SearchResult, numQ)
	for i, q := range queries {
		bruteResults[i] = BruteForceSearch(store, q, k, distFn)
	}

	for _, ef := range []int{10, 20, 50, 100, 200, 400} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			var totalRecall float64
			for qi, q := range queries {
				results := h.SearchWithEf(q, k, ef)
				bfSet := make(map[int32]bool)
				for _, r := range bruteResults[qi] {
					bfSet[r.ID] = true
				}
				hits := 0
				for _, r := range results {
					if bfSet[r.ID] {
						hits++
					}
				}
				totalRecall += float64(hits) / float64(k)
			}
			avgRecall := totalRecall / float64(numQ)
			b.ReportMetric(avgRecall, "recall")
			b.ResetTimer()
			for range b.N {
				qi := qiFromN(b.N, numQ)
				h.SearchWithEf(queries[qi], k, ef)
			}
		})
	}
}

func qiFromN(n, numQ int) int { return n % numQ }

func BenchmarkBruteForce_vs_HNSW(b *testing.B) {
	dim := 128
	n := 10_000
	k := 10

	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
	store := newVectorStore(dim, n)
	for i := 0; i < n; i++ {
		v := makeRandomVec(dim, int64(i))
		h.Add(int32(i), v)
		store.add(v)
	}
	distFn := selectDistFn(distance.L2)
	query := makeRandomVec(dim, 99999)

	b.Run("HNSW_ef50", func(b *testing.B) {
		for range b.N {
			h.SearchWithEf(query, k, 50)
		}
	})
	b.Run("HNSW_ef200", func(b *testing.B) {
		for range b.N {
			h.SearchWithEf(query, k, 200)
		}
	})
	b.Run("BruteForce", func(b *testing.B) {
		for range b.N {
			BruteForceSearch(store, query, k, distFn)
		}
	})
}

func BenchmarkHNSW_ConcurrentSearch(b *testing.B) {
	dim := 128
	n := 10_000
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
	for i := 0; i < n; i++ {
		h.Add(int32(i), makeRandomVec(dim, int64(i)))
	}
	query := makeRandomVec(dim, 99999)

	for _, g := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("goroutines=%d", g), func(b *testing.B) {
			perG := b.N / g
			if perG == 0 {
				perG = 1
			}
			b.ResetTimer()
			var wg sync.WaitGroup
			for i := 0; i < g; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for j := 0; j < perG; j++ {
						h.Search(query, 10)
					}
				}()
			}
			wg.Wait()
		})
	}
}
