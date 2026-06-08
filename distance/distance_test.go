package distance

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

var scalarDotProductTests = []struct {
	name     string
	a, b     []float32
	expected float32
}{
	{"empty", []float32{}, []float32{}, 0},
	{"single", []float32{3}, []float32{4}, 12},
	{"dim4_aligned", []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 70},
	{"dim8_aligned", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 120},
	{"dim7_tail", []float32{1, 2, 3, 4, 5, 6, 7}, []float32{7, 6, 5, 4, 3, 2, 1}, 84},
	{"zeros", []float32{0, 0, 0, 0}, []float32{1, 2, 3, 4}, 0},
	{"negative", []float32{-1, -2, -3, -4}, []float32{1, 2, 3, 4}, -30},
}

func TestScalarDotProduct(t *testing.T) {
	for _, tt := range scalarDotProductTests {
		t.Run(tt.name, func(t *testing.T) {
			got := dotProductScalar(tt.a, tt.b)
			if got != tt.expected {
				t.Errorf("dotProductScalar(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.expected)
			}
		})
	}
}

var scalarL2Tests = []struct {
	name     string
	a, b     []float32
	expected float32
}{
	{"single", []float32{3}, []float32{0}, 3},
	{"dim4_aligned", []float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, float32(math.Sqrt(2))},
	{"zero_dist", []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0},
	{"dim3_tail", []float32{1, 2, 3}, []float32{4, 5, 6}, float32(math.Sqrt(27))},
}

func TestScalarL2Squared(t *testing.T) {
	for _, tt := range scalarL2Tests {
		t.Run(tt.name, func(t *testing.T) {
			got := l2SquaredScalar(tt.a, tt.b)
			want := tt.expected * tt.expected
			if math.Abs(float64(got-want)) > 1e-3 {
				t.Errorf("l2SquaredScalar(%v, %v) = %f, want %f", tt.a, tt.b, got, want)
			}
		})
	}
}

func TestScalarCosine(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	got := cosineScalar(a, a)
	if math.Abs(float64(got-1.0)) > 1e-6 {
		t.Errorf("cosineScalar(a, a) = %f, want 1.0", got)
	}
	b := []float32{1, 0}
	c := []float32{0, 1}
	got = cosineScalar(b, c)
	if math.Abs(float64(got)) > 1e-6 {
		t.Errorf("cosineScalar(%v, %v) = %f, want 0.0", b, c, got)
	}
	d := []float32{1, 2, 3}
	got = cosineScalar(d, []float32{-1, -2, -3})
	if math.Abs(float64(got+1.0)) > 1e-6 {
		t.Errorf("cosineScalar(d, -d) = %f, want -1.0", got)
	}
}

func TestScalarNormalize(t *testing.T) {
	v := []float32{3, 4}
	normScalar(v)
	got := v[0]*v[0] + v[1]*v[1]
	if math.Abs(float64(got-1.0)) > 1e-6 {
		t.Errorf("after Normalize, ||v||^2 = %f, want 1.0", got)
	}
}

func TestScalarNormalizeZero(t *testing.T) {
	v := []float32{0, 0, 0}
	normScalar(v)
	if math.IsNaN(float64(v[0])) {
		t.Error("Normalize on zero vector produced NaN")
	}
}

func makeRandomVec(n int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), uint64(seed)*7+1))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func TestDotProduct_MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256, 768, 1536} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := makeRandomVec(dim, 123)
			got := DotProduct(a, b)
			want := dotProductScalar(a, b)
			if math.Abs(float64(got-want)) > float64(math.Abs(float64(want)))*1e-4+1e-6 {
				t.Errorf("DotProduct(dim=%d) = %f, scalar = %f", dim, got, want)
			}
		})
	}
}

func TestL2Squared_MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 4, 7, 8, 15, 16, 64, 256, 768, 1536} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := makeRandomVec(dim, 123)
			got := L2Squared(a, b)
			want := l2SquaredScalar(a, b)
			if math.Abs(float64(got-want)) > float64(math.Abs(float64(want)))*1e-4+1e-6 {
				t.Errorf("L2Squared(dim=%d) = %f, scalar = %f", dim, got, want)
			}
		})
	}
}

func TestCosineSimilarity_MatchesScalar(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 64, 256, 768, 1536} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := makeRandomVec(dim, 123)
			got := CosineSimilarity(a, b)
			want := cosineScalar(a, b)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("CosineSimilarity(dim=%d) = %f, scalar = %f", dim, got, want)
			}
		})
	}
}

func TestNormalize_MatchesScalar(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 64, 256, 768, 1536} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			a := makeRandomVec(dim, 42)
			b := make([]float32, dim)
			copy(b, a)
			Normalize(a)
			normScalar(b)
			for i := range a {
				if math.Abs(float64(a[i]-b[i])) > 1e-5 {
					t.Errorf("Normalize(dim=%d)[%d] = %f, scalar = %f", dim, i, a[i], b[i])
					break
				}
			}
		})
	}
}

func TestNormalize_ZeroVector(t *testing.T) {
	v := make([]float32, 16)
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Errorf("Normalize(zero)[%d] = %f, want 0", i, x)
		}
	}
}

func TestDistanceFunctions_LargeDim(t *testing.T) {
	dim := 4096
	a := makeRandomVec(dim, 42)
	b := makeRandomVec(dim, 123)
	got := DotProduct(a, b)
	want := dotProductScalar(a, b)
	relErr := math.Abs(float64(got-want)) / (math.Abs(float64(want)) + 1e-10)
	if relErr > 1e-3 {
		t.Errorf("DotProduct(dim=%d) relErr = %e, got=%f want=%f", dim, relErr, got, want)
	}
}

func TestDistanceFunctions_SingleElement(t *testing.T) {
	a := []float32{3.5}
	b := []float32{2.0}
	if got := DotProduct(a, b); math.Abs(float64(got-7.0)) > 1e-6 {
		t.Errorf("DotProduct([3.5],[2.0]) = %f, want 7.0", got)
	}
	if got := L2Distance(a, b); math.Abs(float64(got-1.5)) > 1e-6 {
		t.Errorf("L2Distance([3.5],[2.0]) = %f, want 1.5", got)
	}
}

func BenchmarkDotProduct_Dim(b *testing.B) {
	for _, dim := range []int{128, 256, 384, 768, 1536} {
		a := makeRandomVec(dim, 1)
		c := makeRandomVec(dim, 2)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = DotProduct(a, c)
			}
		})
	}
}

func BenchmarkCosineSimilarity_Dim(b *testing.B) {
	for _, dim := range []int{128, 384, 768, 1536} {
		a := makeRandomVec(dim, 1)
		c := makeRandomVec(dim, 2)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = CosineSimilarity(a, c)
			}
		})
	}
}

func BenchmarkL2Distance_Dim(b *testing.B) {
	for _, dim := range []int{128, 384, 768, 1536} {
		a := makeRandomVec(dim, 1)
		c := makeRandomVec(dim, 2)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = L2Distance(a, c)
			}
		})
	}
}

func BenchmarkNormalize_Dim(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		v := makeRandomVec(dim, 1)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				Normalize(v)
			}
		})
	}
}
