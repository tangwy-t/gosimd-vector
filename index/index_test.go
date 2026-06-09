package index

import (
	"math"
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

func TestVectorStore_AddAndGet(t *testing.T) {
	store := newVectorStore(3, 8)
	idx0, err := store.add([]float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	idx1, err := store.add([]float32{4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	if idx0 != 0 || idx1 != 1 {
		t.Errorf("expected indices 0,1, got %d,%d", idx0, idx1)
	}

	v := store.get(0)
	if len(v) != 3 || v[0] != 1 || v[1] != 2 || v[2] != 3 {
		t.Errorf("get(0) = %v, want [1 2 3]", v)
	}

	_, err = store.add([]float32{1, 2})
	if err == nil {
		t.Error("expected error on dimension mismatch")
	}
}

func TestVectorStore_Delete(t *testing.T) {
	store := newVectorStore(2, 4)
	store.add([]float32{1, 0})
	store.add([]float32{0, 1})
	store.remove(0)

	if !store.isDeleted(0) {
		t.Error("ID 0 should be deleted")
	}
	if store.isDeleted(1) {
		t.Error("ID 1 should NOT be deleted")
	}
}

func TestBruteForceSearch_L2(t *testing.T) {
	store := newVectorStore(2, 16)
	store.add([]float32{0, 0})
	store.add([]float32{1, 0})
	store.add([]float32{0, 1})
	store.add([]float32{1, 1})

	distFn := selectDistFn(distance.L2)
	results := BruteForceSearch(store, []float32{0, 0}, 2, distFn)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// ID=0 is at distance 0, should be first
	if results[0].ID != 0 {
		t.Errorf("expected ID=0 first, got ID=%d dist=%f", results[0].ID, results[0].Distance)
	}
	if results[0].Distance != 0 {
		t.Errorf("expected dist=0 for ID=0, got %f", results[0].Distance)
	}
}

func TestBruteForceSearch_Cosine(t *testing.T) {
	store := newVectorStore(2, 16)
	a := []float32{1, 0}
	b := []float32{0, 1}
	c := []float32{1, 1}
	store.add(a)
	store.add(b)
	store.add(c)

	distFn := selectDistFn(distance.Cosine)
	query := []float32{1, 0}
	results := BruteForceSearch(store, query, 3, distFn)

	// ID=0 is same direction → dist=0
	if results[0].ID != 0 {
		t.Errorf("expected ID=0 first, got ID=%d dist=%f", results[0].ID, results[0].Distance)
	}
	if math.Abs(float64(results[0].Distance)) > 1e-6 {
		t.Errorf("expected dist≈0 for same direction, got %f", results[0].Distance)
	}
}

func TestMinHeap(t *testing.T) {
	h := newMinHeap()
	h.Push(candidate{id: 3, dist: 10})
	h.Push(candidate{id: 1, dist: 2})
	h.Push(candidate{id: 2, dist: 5})

	c := h.Pop()
	if c.id != 1 || c.dist != 2 {
		t.Errorf("expected {1, 2}, got {%d, %f}", c.id, c.dist)
	}
}

func TestMaxHeap(t *testing.T) {
	h := newMaxHeap()
	h.Push(candidate{id: 3, dist: 10})
	h.Push(candidate{id: 1, dist: 2})
	h.Push(candidate{id: 2, dist: 5})

	c := h.Pop()
	if c.id != 3 || c.dist != 10 {
		t.Errorf("expected {3, 10}, got {%d, %f}", c.id, c.dist)
	}
}
