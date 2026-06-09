package search

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

func makeRandomVecBench(dim int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), 0))
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.Float64()*2 - 1)
	}
	return v
}

func BenchmarkEngine_Build(b *testing.B) {
	for _, dim := range []int{128, 384, 768} {
		for _, n := range []int{1_000, 10_000} {
			b.Run(fmt.Sprintf("n=%d_dim=%d", n, dim), func(b *testing.B) {
				vecs := make([][]float32, n)
				for i := range vecs {
					vecs[i] = makeRandomVecBench(dim, int64(i))
				}
				b.ResetTimer()
				for range b.N {
					e := New(dim, distance.L2, WithM(16), WithEfConstruction(200))
					for i, v := range vecs {
						e.Add(int32(i), v)
					}
				}
			})
		}
	}
}

func BenchmarkEngine_Search(b *testing.B) {
	dim := 768
	for _, n := range []int{10_000, 100_000} {
		e := New(dim, distance.L2, WithM(16), WithEfConstruction(200))
		for i := 0; i < n; i++ {
			e.Add(int32(i), makeRandomVecBench(dim, int64(i)))
		}
		query := makeRandomVecBench(dim, 99999)
		b.Run(fmt.Sprintf("n=%d_k=10", n), func(b *testing.B) {
			for range b.N {
				e.Search(query, 10)
			}
		})
	}
}

func BenchmarkEngine_CosineSearch(b *testing.B) {
	dim := 768
	n := 10_000
	e := New(dim, distance.Cosine, WithM(16), WithEfConstruction(200))
	for i := 0; i < n; i++ {
		e.Add(int32(i), makeRandomVecBench(dim, int64(i)))
	}
	query := makeRandomVecBench(dim, 99999)
	b.ResetTimer()
	for range b.N {
		e.Search(query, 10)
	}
}
