package distance

import (
	"math"
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
