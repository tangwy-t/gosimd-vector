package index

import (
	"math/rand/v2"
	"testing"

	"github.com/tangwy-t/gosimd-vector/distance"
)

func TestHNSW_Add_Basic(t *testing.T) {
	h := NewHNSW(Config{Dim: 4, Metric: distance.L2, M: 8, EfConstruction: 50})
	h.Add(0, []float32{1, 2, 3, 4})
	h.Add(1, []float32{1, 2, 3, 5})
	h.Add(2, []float32{5, 6, 7, 8})
	if h.Count() != 3 {
		t.Errorf("want 3, got %d", h.Count())
	}
}

func TestHNSW_Search_L2(t *testing.T) {
	h := NewHNSW(Config{Dim: 4, Metric: distance.L2, M: 16, EfConstruction: 200})
	h.Add(0, []float32{0, 0, 0, 0})
	h.Add(1, []float32{1, 0, 0, 0})
	h.Add(2, []float32{10, 10, 10, 10})
	h.Add(3, []float32{0, 1, 0, 0})
	results := h.Search([]float32{0.1, 0.1, 0, 0}, 2)
	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != 0 && results[0].ID != 3 {
		t.Errorf("top-2 should include ID 0 or 3; got %v", results)
	}
}

func TestHNSW_Search_RecallVsBruteForce(t *testing.T) {
	dim := 64
	n := 2000
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 16, EfConstruction: 200})
	store := newVectorStore(dim, n)
	for i := 0; i < n; i++ {
		v := makeRandomVec(dim, int64(i))
		h.Add(int32(i), v)
		store.add(v)
	}
	distFn := selectDistFn(distance.L2)
	query := makeRandomVec(dim, 9999)
	bfResults := BruteForceSearch(store, query, 10, distFn)
	hnswResults := h.Search(query, 10)
	hits := 0
	bfSet := make(map[int32]bool)
	for _, r := range bfResults {
		bfSet[r.ID] = true
	}
	for _, r := range hnswResults {
		if bfSet[r.ID] {
			hits++
		}
	}
	recall := float64(hits) / 10.0
	if recall < 0.7 {
		t.Errorf("recall=%.2f < 0.7, got %v vs brute %v", recall, hnswResults, bfResults)
	}
}

func TestHNSW_Remove(t *testing.T) {
	h := NewHNSW(Config{Dim: 4, Metric: distance.L2, M: 8, EfConstruction: 50})
	h.Add(0, []float32{0, 0, 0, 0})
	h.Add(1, []float32{0.01, 0, 0, 0})
	h.Add(2, []float32{10, 10, 10, 10})
	h.Remove(0)
	if h.Count() != 2 {
		t.Errorf("want 2, got %d", h.Count())
	}
	results := h.Search([]float32{0, 0, 0, 0}, 10)
	for _, r := range results {
		if r.ID == 0 {
			t.Error("ID 0 should be removed")
		}
	}
}

func makeRandomVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), 0))
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.Float64()*2 - 1)
	}
	return v
}
