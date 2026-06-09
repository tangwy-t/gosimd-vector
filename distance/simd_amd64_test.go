//go:build amd64

package distance

import (
	"fmt"
	"math"
	"testing"
)

func TestCosineFused_AVX2_MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 4, 7, 8, 15, 16, 64, 256, 768, 1536, 4096} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := makeRandomVec(dim, 123)
			got := cosineFusedAVX2(a, b)
			want := cosineScalar(a, b)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("cosineFusedAVX2(dim=%d) = %f, scalar = %f", dim, got, want)
			}
		})
	}
}

func TestCosineFused_AVX512_MatchesScalar(t *testing.T) {
	if !hasAVX512 {
		t.Skip("AVX-512 not available")
	}
	for _, dim := range []int{1, 4, 15, 16, 31, 64, 256, 768, 1536, 4096} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := makeRandomVec(dim, 123)
			got := cosineFusedAVX512(a, b)
			want := cosineScalar(a, b)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("cosineFusedAVX512(dim=%d) = %f, scalar = %f", dim, got, want)
			}
		})
	}
}

func TestCosineFused_ZeroVector(t *testing.T) {
	a := make([]float32, 16)
	b := makeRandomVec(16, 42)
	if got := cosineFusedAVX2(a, b); got != 0 {
		t.Errorf("cosineFused(zero, b) = %f, want 0", got)
	}
}

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

func BenchmarkCosineSimilarity_Dim768(b *testing.B) {
	dim := 768
	a := makeRandomVec(dim, 1)
	c := makeRandomVec(dim, 2)
	b.Run("Old_3pass_AVX2", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = cosineAVX2(a, c)
		}
	})
	b.Run("Fused_1pass_AVX2", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = cosineFusedAVX2(a, c)
		}
	})
	if hasAVX512 {
		b.Run("Old_3pass_AVX512", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = cosineAVX512(a, c)
			}
		})
		b.Run("Fused_1pass_AVX512", func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = cosineFusedAVX512(a, c)
			}
		})
	}
	b.Run("Scalar", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = cosineScalar(a, c)
		}
	})
}
