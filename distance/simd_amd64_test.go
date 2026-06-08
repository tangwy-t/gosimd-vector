//go:build amd64

package distance

import "testing"

func BenchmarkDotProduct_Implementations(b *testing.B) {
	dim := 768
	a := makeRandomVec(dim, 1)
	c := makeRandomVec(dim, 2)
	b.Run("Scalar", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = dotProductScalar(a, c)
		}
	})
	if hasAVX2 {
		b.Run("AVX2", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = dotProductAVX2(a, c)
			}
		})
	}
	if hasAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = dotProductAVX512(a, c)
			}
		})
	}
}

func BenchmarkCosineSimilarity_Implementations(b *testing.B) {
	dim := 768
	a := makeRandomVec(dim, 1)
	c := makeRandomVec(dim, 2)
	b.Run("Scalar", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = cosineScalar(a, c)
		}
	})
	if hasAVX2 {
		b.Run("AVX2", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = cosineAVX2(a, c)
			}
		})
	}
	if hasAVX512 {
		b.Run("AVX512", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = cosineAVX512(a, c)
			}
		})
	}
}
