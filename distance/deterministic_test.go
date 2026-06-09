package distance

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestDeterministicAllImplementations verifies that scalar, AVX2, AVX512,
// and fused cosine implementations produce mathematically identical results
// on a wide range of inputs.
func TestDeterministicAllImplementations(t *testing.T) {
	dims := []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 512, 768, 1024}
	
	rng := rand.New(rand.NewSource(42))
	
	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			// Generate random vectors
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := 0; i < dim; i++ {
				a[i] = float32(rng.NormFloat64())
				b[i] = float32(rng.NormFloat64())
			}
			
			// Test dot product
			scalarDot := dotProductScalar(a, b)
			avx2Dot := dotProductAVX2(a, b)
			avx512Dot := dotProductAVX512(a, b)
			
			// Use relative tolerance (1e-4) due to floating-point accumulation order differences
			// SIMD uses parallel lanes then sums them, scalar is sequential - different rounding paths
			const relTol = 1e-4
			if scalarDot != 0 {
				avx2RelErr := math.Abs(float64(scalarDot-avx2Dot)) / math.Abs(float64(scalarDot))
				if avx2RelErr > relTol {
					t.Errorf("dim%d: scalar dot (%f) vs AVX2 dot (%f), relative error=%e > %e", 
						dim, scalarDot, avx2Dot, avx2RelErr, relTol)
				}
				avx512RelErr := math.Abs(float64(scalarDot-avx512Dot)) / math.Abs(float64(scalarDot))
				if avx512RelErr > relTol {
					t.Errorf("dim%d: scalar dot (%f) vs AVX512 dot (%f), relative error=%e > %e", 
						dim, scalarDot, avx512Dot, avx512RelErr, relTol)
				}
			} else {
				// For zero dot product, use absolute tolerance
				const absTol = 1e-5
				if math.Abs(float64(avx2Dot)) > absTol {
					t.Errorf("dim%d: scalar dot=0 but AVX2 dot=%f", dim, avx2Dot)
				}
				if math.Abs(float64(avx512Dot)) > absTol {
					t.Errorf("dim%d: scalar dot=0 but AVX512 dot=%f", dim, avx512Dot)
				}
			}
			
			// Test cosine similarity (fused vs old 3-pass)
			scalarCos := cosineScalar(a, b)
			oldCos := cosineAVX2(a, b)
			fusedCos := cosineFusedAVX2(a, b)
			fusedCos512 := cosineFusedAVX512(a, b)
			
			const tolerance = 1e-5
			if math.Abs(float64(scalarCos-oldCos)) > tolerance {
				t.Errorf("dim%d: scalar cosine (%f) != old AVX2 cosine (%f), diff=%e",
					dim, scalarCos, oldCos, scalarCos-oldCos)
			}
			if math.Abs(float64(scalarCos-fusedCos)) > tolerance {
				t.Errorf("dim%d: scalar cosine (%f) != fused AVX2 cosine (%f), diff=%e",
					dim, scalarCos, fusedCos, scalarCos-fusedCos)
			}
			if math.Abs(float64(scalarCos-fusedCos512)) > tolerance {
				t.Errorf("dim%d: scalar cosine (%f) != fused AVX512 cosine (%f), diff=%e",
					dim, scalarCos, fusedCos512, scalarCos-fusedCos512)
			}
		})
	}
}

// TestEdgeCasesAllImplementations verifies behavior on edge cases
func TestEdgeCasesAllImplementations(t *testing.T) {
	// Zero vectors
	zero := make([]float32, 768)
	one := make([]float32, 768)
	for i := range one {
		one[i] = 1.0
	}
	
	t.Run("zero_vectors", func(t *testing.T) {
		s := dotProductScalar(zero, zero)
		a2 := dotProductAVX2(zero, zero)
		a5 := dotProductAVX512(zero, zero)
		if s != 0 || a2 != 0 || a5 != 0 {
			t.Errorf("zero⋅zero should be 0, got scalar=%f avx2=%f avx512=%f", s, a2, a5)
		}
	})
	
	t.Run("identical_vectors", func(t *testing.T) {
		sc := cosineScalar(one, one)
		fa2 := cosineFusedAVX2(one, one)
		fa5 := cosineFusedAVX512(one, one)
		if sc != 1.0 || fa2 != 1.0 || fa5 != 1.0 {
			t.Errorf("cosine(v,v) should be 1.0, got scalar=%f fusedAVX2=%f fusedAVX512=%f", 
				sc, fa2, fa5)
		}
	})
	
	t.Run("normalized_orthogonal", func(t *testing.T) {
		a := make([]float32, 768)
		b := make([]float32, 768)
		a[0] = 1.0
		b[1] = 1.0
		// Already normalized, should be orthogonal (cosine ≈ 0)
		sc := cosineScalar(a, b)
		fa2 := cosineFusedAVX2(a, b)
		fa5 := cosineFusedAVX512(a, b)
		const tol = 1e-6
		if math.Abs(float64(sc)) > tol || math.Abs(float64(fa2)) > tol || math.Abs(float64(fa5)) > tol {
			t.Errorf("orthogonal vectors should have cosine ≈ 0, got scalar=%f fusedAVX2=%f fusedAVX512=%f",
				sc, fa2, fa5)
		}
	})
}
