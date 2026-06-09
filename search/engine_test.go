package search

import (
	"bytes"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/tangwy-t/gosimd-vector/distance"
)

func makeRandomVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), 0))
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.Float64()*2 - 1)
	}
	return v
}

func TestEngine_Basic(t *testing.T) {
	e := New(16, distance.L2, WithM(16), WithEfConstruction(200))
	for i := 0; i < 100; i++ {
		if err := e.Add(int32(i), makeRandomVec(16, int64(i))); err != nil {
			t.Fatal(err)
		}
	}
	if e.Count() != 100 {
		t.Fatalf("want 100, got %d", e.Count())
	}
	results := e.Search(makeRandomVec(16, 42), 5)
	if len(results) != 5 {
		t.Fatalf("want 5, got %d", len(results))
	}
}

func TestEngine_CosineNormalization(t *testing.T) {
	e := New(8, distance.Cosine, WithM(16), WithEfConstruction(200))
	for i := 0; i < 50; i++ {
		v := makeRandomVec(8, int64(i))
		if err := e.Add(int32(i), v); err != nil {
			t.Fatal(err)
		}
	}
	if e.Count() != 50 {
		t.Fatalf("want 50, got %d", e.Count())
	}
	results := e.Search(makeRandomVec(8, 999), 5)
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	for _, r := range results {
		if r.Distance < -0.01 || r.Distance > 2.01 {
			t.Errorf("cosine distance %f out of expected range [0,2]", r.Distance)
		}
	}
}

func TestEngine_AddBatch(t *testing.T) {
	e := New(8, distance.L2)
	batch := make(map[int32][]float32, 50)
	for i := 0; i < 50; i++ {
		batch[int32(i)] = makeRandomVec(8, int64(i))
	}
	if err := e.AddBatch(batch); err != nil {
		t.Fatal(err)
	}
	if e.Count() != 50 {
		t.Fatalf("want 50, got %d", e.Count())
	}
}

func TestEngine_SaveLoad(t *testing.T) {
	e := New(8, distance.L2, WithM(8), WithEfConstruction(50))
	for i := 0; i < 100; i++ {
		if err := e.Add(int32(i), makeRandomVec(8, int64(i))); err != nil {
			t.Fatal(err)
		}
	}
	originalCount := e.Count()

	var buf bytes.Buffer
	if err := e.Save(&buf); err != nil {
		t.Fatal(err)
	}

	e2 := New(8, distance.L2)
	if err := e2.Load(&buf); err != nil {
		t.Fatal(err)
	}
	if e2.Count() != originalCount {
		t.Fatalf("count mismatch: %d vs %d", e2.Count(), originalCount)
	}

	query := makeRandomVec(8, 5555)
	r1 := e.Search(query, 5)
	r2 := e2.Search(query, 5)
	if len(r1) != len(r2) {
		t.Fatalf("result length mismatch: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] ID mismatch: %d vs %d", i, r1[i].ID, r2[i].ID)
		}
	}
}

func TestEngine_ConcurrentSearch(t *testing.T) {
	e := New(16, distance.L2, WithM(16), WithEfConstruction(100))
	for i := 0; i < 200; i++ {
		if err := e.Add(int32(i), makeRandomVec(16, int64(i))); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				q := makeRandomVec(16, seed*1000+int64(i))
				results := e.Search(q, 5)
				if len(results) == 0 {
					t.Errorf("goroutine seed=%d iter=%d: no results", seed, i)
				}
			}
		}(int64(g))
	}
	wg.Wait()
}

func TestEngine_Remove(t *testing.T) {
	e := New(8, distance.L2)
	for i := 0; i < 20; i++ {
		if err := e.Add(int32(i), makeRandomVec(8, int64(i))); err != nil {
			t.Fatal(err)
		}
	}
	e.Remove(0)
	e.Remove(1)
	if e.Count() != 18 {
		t.Fatalf("want 18, got %d", e.Count())
	}
	results := e.Search(makeRandomVec(8, 42), 20)
	for _, r := range results {
		if r.ID == 0 || r.ID == 1 {
			t.Errorf("removed ID %d should not appear", r.ID)
		}
	}
}
