package index

import (
	"bytes"
	"testing"

	"github.com/tangwy-t/gosimd-vector/distance"
)

func TestHNSW_Serialize_RoundTrip(t *testing.T) {
	dim := 8
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 8, EfConstruction: 50})
	for i := 0; i < 100; i++ {
		h.Add(int32(i), makeRandomVec(dim, int64(i)))
	}
	originalCount := h.Count()

	var buf bytes.Buffer
	err := h.Save(&buf)
	if err != nil {
		t.Fatal(err)
	}

	h2, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if h2.Count() != originalCount {
		t.Errorf("count mismatch: %d vs %d", h2.Count(), originalCount)
	}

	query := makeRandomVec(dim, 9999)
	r1 := h.Search(query, 5)
	r2 := h2.Search(query, 5)
	if len(r1) != len(r2) {
		t.Fatalf("length mismatch: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] mismatch: ID %d vs %d", i, r1[i].ID, r2[i].ID)
		}
	}
}

func TestHNSW_Serialize_EmptyIndex(t *testing.T) {
	dim := 4
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 8, EfConstruction: 50})

	var buf bytes.Buffer
	err := h.Save(&buf)
	if err != nil {
		t.Fatal(err)
	}

	h2, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if h2.Count() != 0 {
		t.Errorf("expected 0, got %d", h2.Count())
	}

	results := h2.Search([]float32{1, 2, 3, 4}, 5)
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestHNSW_Serialize_WithRemovals(t *testing.T) {
	dim := 8
	h := NewHNSW(Config{Dim: dim, Metric: distance.L2, M: 8, EfConstruction: 50})
	for i := 0; i < 50; i++ {
		h.Add(int32(i), makeRandomVec(dim, int64(i)))
	}

	h.Remove(0)
	h.Remove(5)
	h.Remove(10)
	originalCount := h.Count()

	var buf bytes.Buffer
	err := h.Save(&buf)
	if err != nil {
		t.Fatal(err)
	}

	h2, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if h2.Count() != originalCount {
		t.Errorf("count mismatch: %d vs %d", h2.Count(), originalCount)
	}

	query := makeRandomVec(dim, 7777)
	r1 := h.Search(query, 10)
	r2 := h2.Search(query, 10)
	if len(r1) != len(r2) {
		t.Fatalf("length mismatch: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] mismatch: ID %d vs %d", i, r1[i].ID, r2[i].ID)
		}
	}

	for _, r := range r2 {
		if r.ID == 0 || r.ID == 5 || r.ID == 10 {
			t.Errorf("removed ID %d should not appear in results", r.ID)
		}
	}
}
