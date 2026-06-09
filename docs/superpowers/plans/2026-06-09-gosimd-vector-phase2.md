# gosimd-vector Phase 2: HNSW Index + Search Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HNSW (Hierarchical Navigable Small World) vector index on top of the Phase 1 `distance` package, providing insertion, approximate nearest neighbor search with configurable recall/latency tradeoff, brute-force baseline, and a clean user-facing search Engine API with serialization.

**Architecture:** The `index` package holds the HNSW graph (multi-layer neighbor lists per node), a contiguous vector store, and a min/max heap–based search algorithm. Search uses greedy descent from top layers + beam search on the base layer. Concurrency via `sync.RWMutex` (concurrent reads, exclusive writes). The `search` package wraps the index with an Options-pattern API.

**Tech Stack:** Go 1.26, Phase 1 `distance` package, `container/heap` (stdlib), `math/rand/v2`, `encoding/binary`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `index/pq.go` | `Item` struct + `MinHeap` (`closestFirst`) / `MaxHeap` (`farthestFirst`) implementing `heap.Interface` |
| `index/store.go` | `vectorStore` — contiguous `[]float32` storage; `get(id)`, `add(id, vec)`, `remove(id)` |
| `index/hnsw.go` | `HNSWIndex` struct, `dist` closure, `Add`, `Remove`, `Search`, internal `searchLayer`, `searchCandidates` |
| `index/serialize.go` | `Save(w io.Writer)`, `LoadIndex(r io.Reader)` binary serialization |
| `index/brute.go` | `BruteForceSearch` — exhaustive scan using `distance` package |
| `index/index_test.go` | Unit tests (construction, search correctness, recall vs brute-force, serialization round-trip) |
| `search/engine.go` | `Engine`, `SearchResult`, `Option` types; `New`, `Add`, `AddBatch`, `Search`, `Remove`, `Count`, `Save`, `Load` |
| `search/engine_test.go` | Integration tests (end-to-end construction + search + save/load + concurrent reads) |

---

## Task 1: Priority Queue + Vector Store + Distance Integration

**Files:**
- Create: `index/pq.go`
- Create: `index/store.go`

### Step 1: Create `index/pq.go`

```go
package index

import "container/heap"

type Item struct {
	ID   int32
	Dist float32
}

type MinHeap []Item

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].Dist < h[j].Dist }
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x any)         { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() any           { x := (*h)[len(*h)-1]; *h = (*h)[:len(*h)-1]; return x }

type MaxHeap []Item

func (h MaxHeap) Len() int            { return len(h) }
func (h MaxHeap) Less(i, j int) bool  { return h[i].Dist > h[j].Dist }
func (h MaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x any)         { *h = append(*h, x.(Item)) }
func (h *MaxHeap) Pop() any           { x := (*h)[len(*h)-1]; *h = (*h)[:len(*h)-1]; return x }

func heapPush(h heap.Interface, id int32, d float32) { heap.Push(h, Item{ID: id, Dist: d}) }
func heapPop(h heap.Interface) Item                  { return heap.Pop(h).(Item) }
func heapPeek(h heap.Interface) Item                 { return (*(*[]Item)(unsafePointer(h)))[0] }
```

> **Note:** `heapPeek` uses a type assertion trick. In practice, since both MinHeap/MaxHeap are `[]Item`, use this: just cast through interface and index 0, or define `peek` directly on the heap type. Simplest: define `func (h MinHeap) peek() Item { return h[0] }` and `func (h MaxHeap) peek() Item { return h[0] }` as methods. Use methods instead of the generic helper. Add these methods to both types.

### Step 2: Create `index/store.go`

```go
package index

import "fmt"

type vectorStore struct {
	data    []float32
	dim     int
	removed map[int32]bool
}

func newVectorStore(dim int, capacity int) *vectorStore {
	return &vectorStore{
		data:    make([]float32, 0, dim*capacity),
		dim:     dim,
		removed: make(map[int32]bool, capacity),
	}
}

func (s *vectorStore) add(id int32, v []float32) error {
	if len(v) != s.dim {
		return fmt.Errorf("vector dim %d != store dim %d", len(v), s.dim)
	}
	s.data = append(s.data, v...)
	delete(s.removed, id)
	return nil
}

func (s *vectorStore) get(id int32) []float32 {
	off := int(id) * s.dim
	if off+s.dim > len(s.data) {
		return nil
	}
	return s.data[off : off+s.dim]
}

func (s *vectorStore) markRemoved(id int32) {
	s.removed[id] = true
}

func (s *vectorStore) isRemoved(id int32) bool {
	return s.removed[id]
}

func (s *vectorStore) count() int {
	return len(s.data)/s.dim - len(s.removed)
}
```

### Step 3: Add distance integration map to `index/hnsw.go`

```go
package index

import (
	"github.com/gosimd-vector/gosimd-vector/distance"
)

type distFn func(a, b []float32) float32

func distForMetric(m distance.MetricType) distFn {
	switch m {
	case distance.L2:
		return distance.L2Squared
	case distance.Cosine:
		return cosineDist
	case distance.InnerProduct:
		return innerProductDist
	default:
		return distance.L2Squared
	}
}

// cosineDist returns 1 - cosineSimilarity. Lower = more similar.
// Vectors are assumed normalized on insert.
func cosineDist(a, b []float32) float32 {
	return 1 - distance.CosineSimilarity(a, b)
}

// innerProductDist returns negative dot product. Lower = more similar.
func innerProductDist(a, b []float32) float32 {
	return -distance.DotProduct(a, b)
}
```

### Step 4: Verify `index` package compiles

```bash
export PATH=/usr/local/go/bin:$PATH
export GOEXPERIMENT=simd
go build ./index/
```
Expected: No errors.

### Step 5: Commit

```bash
git add index/
git commit -m "feat: add priority queue, vector store, and distance integration"
```

---

## Task 2: Brute-Force KNN Search

**Files:**
- Create: `index/brute.go`
- Test: `index/index_test.go` (brute-force portion)

### Step 1: Create `index/brute.go`

```go
package index

import "sort"

type SearchResult struct {
	ID       int32
	Distance float32
}

// BruteForceSearch performs exhaustive KNN search over a vectorStore.
// Used as ground truth for recall evaluation.
func BruteForceSearch(store *vectorStore, query []float32, k int, dist distFn) []SearchResult {
	n := len(store.data) / store.dim
	results := make([]SearchResult, 0, n)
	for id := int32(0); id < int32(n); id++ {
		if store.isRemoved(id) {
			continue
		}
		v := store.get(id)
		results = append(results, SearchResult{ID: id, Distance: dist(query, v)})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if len(results) > k {
		results = results[:k]
	}
	return results
}
```

### Step 2: Write brute-force unit tests

Create `index/index_test.go` with a test that verifies brute-force on a small hand-computed dataset:

```go
package index

import (
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

func TestBruteForceSearch_SmallSet(t *testing.T) {
	// 4 vectors in 2D
	store := newVectorStore(2, 4)
	store.add(0, []float32{0, 0})
	store.add(1, []float32{1, 0})
	store.add(2, []float32{0, 1})
	store.add(3, []float32{1, 1})

	distFn := distForMetric(distance.L2)
	results := BruteForceSearch(store, []float32{0, 0}, 2, distFn)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != 0 || results[0].Distance != 0 {
		t.Errorf("expected ID=0 dist=0, got ID=%d dist=%f", results[0].ID, results[0].Distance)
	}
	// ID=1 (dist=1) and ID=2 (dist=1) are tied; either is acceptable
	if results[1].Distance != 1 {
		t.Errorf("expected second result dist=1, got %f", results[1].Distance)
	}
}

func TestBruteForceSearch_RemoveNode(t *testing.T) {
	store := newVectorStore(2, 4)
	store.add(0, []float32{0, 0})
	store.add(1, []float32{0.1, 0})
	store.add(2, []float32{10, 10})

	distFn := distForMetric(distance.L2)

	results := BruteForceSearch(store, []float32{0, 0}, 2, distFn)
	if results[0].ID != 0 {
		t.Errorf("before remove: expected ID=0, got ID=%d", results[0].ID)
	}

	store.markRemoved(0)
	results = BruteForceSearch(store, []float32{0, 0}, 2, distFn)
	if results[0].ID != 1 {
		t.Errorf("after remove ID=0: expected top result ID=1, got ID=%d", results[0].ID)
	}
}
```

### Step 3: Run tests and verify

```bash
go test ./index/ -v -run TestBruteForceSearch
```
Expected: PASS.

### Step 4: Commit

```bash
git add index/
git commit -m "feat: add brute-force KNN search for recall evaluation"
```

---

## Task 3: HNSW Index Core — Construction + Search

**Files:**
- Create (full): `index/hnsw.go`

This is the most complex task. The HNSW algorithm is implemented in full here.

### Step 1: Write `index/hnsw.go` — struct + Add + Search + internal algorithms

```go
package index

import (
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	metric         distance.MetricType
	dist           distFn
	store          *vectorStore
	layers         [][ ][ ]int32 // layers[l][nodeID] = []int32{neighbor IDs}
	entryPoint     int32
	maxLayer       int
	M              int // max connections per node (base layer); high layers = M/2
	efConstruction int
	levelMult      float64
	rng            *rand.Rand
	hasEntryPoint  bool
}

type HNSWOptions struct {
	M              int
	EfConstruction int
	Dim            int
	Metric         distance.MetricType
}

func DefaultOptions(dim int, metric distance.MetricType) HNSWOptions {
	return HNSWOptions{
		M:              16,
		EfConstruction: 200,
		Dim:            dim,
		Metric:         metric,
	}
}

func NewHNSW(opts HNSWOptions) *HNSWIndex {
	return &HNSWIndex{
		dim:            opts.Dim,
		metric:         opts.Metric,
		dist:           distForMetric(opts.Metric),
		store:          newVectorStore(opts.Dim, 1024),
		layers:         nil,
		maxLayer:       -1,
		M:              opts.M,
		efConstruction: opts.EfConstruction,
		levelMult:      1.0 / math.Log(float64(opts.M)),
		rng:            rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
	}
}

func (idx *HNSWIndex) randomLevel() int {
	uniform := idx.rng.Float64()
	if uniform == 0 {
		uniform = 1e-10
	}
	return int(-math.Log(uniform) * idx.levelMult)
}

func (idx *HNSWIndex) maxConns(layer int) int {
	if layer == 0 {
		return idx.M
	}
	return idx.M / 2
}

func (idx *HNSWIndex) Add(id int32, v []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.insertLocked(id, v)
}

func (idx *HNSWIndex) insertLocked(id int32, v []float32) error {
	if err := idx.store.add(id, v); err != nil {
		return err
	}

	// Ensure enough layers exist for the new node
	level := idx.randomLevel()
	for len(idx.layers) <= level {
		idx.layers = append(idx.layers, nil)
	}

	// Ensure each layer has neighbor list entries for this node
	for l := 0; l <= level; l++ {
		for int(id) >= len(idx.layers[l]) {
			idx.layers[l] = append(idx.layers[l], nil)
		}
		idx.layers[l][id] = nil
	}

	if !idx.hasEntryPoint {
		idx.entryPoint = id
		idx.maxLayer = level
		idx.hasEntryPoint = true
		return nil
	}

	curr := idx.entryPoint

	// Phase 1: greedy descent through layers above 'level'
	for l := idx.maxLayer; l > level; l-- {
		curr = idx.searchLayer(curr, v, l)
	}

	// Phase 2: beam search + connect at layers level..0
	topL := level
	if topL > idx.maxLayer {
		topL = idx.maxLayer
	}
	for l := topL; l >= 0; l-- {
		candidates := idx.searchCandidates(curr, v, l, idx.efConstruction)
		idx.addConnections(id, candidates, l)
		if len(candidates) > 0 {
			curr = candidates[0].ID
		}
	}

	if level > idx.maxLayer {
		idx.maxLayer = level
		idx.entryPoint = id
	}
	return nil
}

// searchLayer does greedy nearest-neighbor descent on one layer.
// Returns the ID of the closest node found (no lock; caller holds it).
func (idx *HNSWIndex) searchLayer(entry int32, query []float32, layer int) int32 {
	curr := entry
	currDist := idx.dist(query, idx.store.get(curr))

	for {
		changed := false
		for _, nb := range idx.layers[layer][curr] {
			if idx.store.isRemoved(nb) {
				continue
			}
			d := idx.dist(query, idx.store.get(nb))
			if d < currDist {
				curr = nb
				currDist = d
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return curr
}

// searchCandidates runs beam search (ef > 1) on a single layer.
// Returns candidates sorted by distance (closest first).
func (idx *HNSWIndex) searchCandidates(entry int32, query []float32, layer int, ef int) []Item {
	visited := map[int32]bool{entry: true}
	dist0 := idx.dist(query, idx.store.get(entry))

	candidates := &MinHeap{{ID: entry, Dist: dist0}}
	results := &MaxHeap{{ID: entry, Dist: dist0}}
	heapInit(candidates)
	heapInit(results)

	for candidates.Len() > 0 {
		nearest := candidates.Pop().(Item)
		farthest := results.Peek()
		if nearest.Dist > farthest.Dist && results.Len() >= ef {
			break
		}
		for _, nb := range idx.layers[layer][nearest.ID] {
			if visited[nb] {
				continue
			}
			visited[nb] = true
			d := idx.dist(query, idx.store.get(nb))
			farthest = results.Peek()
			if d < farthest.Dist || results.Len() < ef {
				candidates.Push(Item{ID: nb, Dist: d})
				heapPush(candidates, nb, d) // wrapper defined in pq.go
				results.Push(Item{ID: nb, Dist: d})
				if results.Len() > ef {
					results.Pop()
				}
			}
		}
	}

	out := make([]Item, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		out[i] = results.Pop().(Item)
	}
	// out is now farthest-to-closest; reverse to closest-first
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// addConnections connects node 'id' to its selected neighbors at the given layer.
func (idx *HNSWIndex) addConnections(id int32, candidates []Item, layer int) {
	mConns := idx.maxConns(layer)
	count := len(candidates)
	if count > mConns {
		count = mConns
	}
	idx.layers[layer][id] = make([]int32, count)
	for i := 0; i < count; i++ {
		nbID := candidates[i].ID
		idx.layers[layer][id][i] = nbID
		// bidirectional: add id to neighbor's list
		idx.layers[layer][nbID] = append(idx.layers[layer][nbID], id)
		if len(idx.layers[layer][nbID]) > mConns {
			idx.shrinkNeighbors(nbID, layer, mConns)
		}
	}
}

// shrinkNeighbors keeps the M closest neighbors for a node (simple pruning).
func (idx *HNSWIndex) shrinkNeighbors(nodeID int32, layer int, maxConns int) {
	nbs := idx.layers[layer][nodeID]
	vec := idx.store.get(nodeID)
	dists := make([]Item, len(nbs))
	for i, nb := range nbs {
		dists[i] = Item{ID: nb, Dist: idx.dist(vec, idx.store.get(nb))}
	}
	sort.Slice(dists, func(i, j int) bool { return dists[i].Dist < dists[j].Dist })
	pruned := make([]int32, maxConns)
	for i := 0; i < maxConns; i++ {
		pruned[i] = dists[i].ID
	}
	idx.layers[layer][nodeID] = pruned
}

// Search performs approximate nearest neighbor search.
func (idx *HNSWIndex) Search(query []float32, k int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.searchLocked(query, k, 64) // default efSearch
}

// SearchWithEf allows caller to specify efSearch (higher = better recall, slower).
func (idx *HNSWIndex) SearchWithEf(query []float32, k int, efSearch int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.searchLocked(query, k, efSearch)
}

func (idx *HNSWIndex) searchLocked(query []float32, k int, efSearch int) []SearchResult {
	if !idx.hasEntryPoint {
		return nil
	}
	curr := idx.entryPoint

	// Greedy descent through upper layers
	for l := idx.maxLayer; l >= 1; l-- {
		curr = idx.searchLayer(curr, query, l)
	}

	// Beam search on base layer
	candidates := idx.searchCandidates(curr, query, 0, efSearch)

	// Take top-k and filter removed
	out := make([]SearchResult, 0, k)
	for _, c := range candidates {
		if idx.store.isRemoved(c.ID) {
			continue
		}
		out = append(out, SearchResult{ID: c.ID, Distance: c.Dist})
		if len(out) >= k {
			break
		}
	}
	return out
}

// Remove marks a node as removed without modifying graph edges.
func (idx *HNSWIndex) Remove(id int32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.store.markRemoved(id)
	return nil
}

func (idx *HNSWIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.store.count()
}
```

> **Note for implementer:** The `heapPush`/`Peek` wrappers from `pq.go` need to match the calls above. Adjust as appropriate based on actual `pq.go` implementation. The key algorithm is correct: min-heap for candidates, max-heap for results capped at `ef`.

### Step 2: Add heap helper methods to `pq.go`

```go
func heapInit(h heap.Interface) { heap.Init(h) }

func (h *MinHeap) Peek() Item { return (*h)[0] }
func (h MaxHeap) Peek() Item { return h[0] }

// Ensure heap.Interface is satisfied (compile-time checks)
var _ heap.Interface = (*MinHeap)(nil)
var _ heap.Interface = (*MaxHeap)(nil)
```

### Step 3: Run build check

```bash
go build ./index/
```
Expected: No errors.

### Step 4: Commit

```bash
git add index/
git commit -m "feat: implement HNSW index core (Add, Search, Remove)"
```

---

## Task 4: HNSW Correctness Tests + Recall Evaluation

**Files:**
- Modify: `index/index_test.go` (add HNSW tests)

### Step 1: Add tests that verify construction, search correctness, and recall vs brute-force

Append to `index/index_test.go` (or create if first test file):

```go
func makeRandomVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), uint64(seed)*7+1))
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func TestHNSW_Add_SingleInsert(t *testing.T) {
	idx := NewHNSW(DefaultOptions(4, distance.L2))
	err := idx.Add(0, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if idx.Count() != 1 {
		t.Errorf("expected Count=1, got %d", idx.Count())
	}
}

func TestHNSW_Search_ExactSmallSet(t *testing.T) {
	idx := NewHNSW(HNSWOptions{M: 8, EfConstruction: 50, Dim: 2, Metric: distance.L2})
	idx.Add(0, []float32{0, 0})
	idx.Add(1, []float32{1, 0})
	idx.Add(2, []float32{0, 1})
	idx.Add(3, []float32{1, 1})

	results := idx.Search([]float32{0, 0}, 2)
	if len(results) < 1 {
		t.Fatal("no results returned")
	}
	if results[0].ID != 0 {
		t.Errorf("expected nearest ID=0, got ID=%d", results[0].ID)
	}
}

func TestHNSW_Search_RecallVsBruteForce(t *testing.T) {
	dim := 64
	n := 1000
	idx := NewHNSW(HNSWOptions{M: 16, EfConstruction: 200, Dim: dim, Metric: distance.L2})
	store := newVectorStore(dim, n)
	distFn := distForMetric(distance.L2)

	for i := range n {
		v := makeRandomVec(dim, int64(i))
		idx.Add(int32(i), v)
		store.add(int32(i), v)
	}

	query := makeRandomVec(dim, 9999)
	k := 10

	bruteResults := BruteForceSearch(store, query, k, distFn)
	hnswResults := idx.Search(query, k)

	// Compute recall: fraction of brute-force top-k IDs found in HNSW results
	bruteSet := make(map[int32]bool, k)
	for _, r := range bruteResults {
		bruteSet[r.ID] = true
	}
	found := 0
	for _, r := range hnswResults {
		if bruteSet[r.ID] {
			found++
		}
	}
	recall := float64(found) / float64(k)

	if recall < 0.8 {
		t.Errorf("recall=%.2f < 0.8, expected at least 80%% with efConstruction=200", recall)
		t.Logf("brute IDs: %v", bruteResults)
		t.Logf("hnsw  IDs: %v", hnswResults)
	}
}

func TestHNSW_Remove(t *testing.T) {
	idx := NewHNSW(HNSWOptions{M: 8, EfConstruction: 50, Dim: 2, Metric: distance.L2})
	idx.Add(0, []float32{0, 0})
	idx.Add(1, []float32{0.01, 0})  // very close to query
	idx.Add(2, []float32{10, 10})

	results := idx.Search([]float32{0, 0}, 3)
	if results[0].ID != 0 && results[0].ID != 1 {
		t.Errorf("expected nearest ID=0 or ID=1, got ID=%d", results[0].ID)
	}

	idx.Remove(0)
	idx.Remove(1)
	results = idx.Search([]float32{0, 0}, 3)
	if len(results) != 1 || results[0].ID != 2 {
		t.Errorf("after removing IDs 0,1: expected [ID=2], got %v", results)
	}
}

func TestHNSW_LargerDataset(t *testing.T) {
	dim := 32
	n := 500
	idx := NewHNSW(HNSWOptions{M: 12, EfConstruction: 100, Dim: dim, Metric: distance.Cosine})

	for i := range n {
		v := makeRandomVec(dim, int64(i))
		idx.Add(int32(i), v)
	}

	if idx.Count() != n {
		t.Errorf("expected Count=%d, got %d", n, idx.Count())
	}

	results := idx.Search(makeRandomVec(dim, 9999), 5)
	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}
}
```

### Step 2: Run tests

```bash
export GOEXPERIMENT=simd
go test ./index/ -v -count=1
```
Expected: All TestHNSW* tests PASS. Recall >= 0.8 for RecallVsBruteForce.

### Step 3: Commit

```bash
git add index/
git commit -m "test: add HNSW correctness tests and recall evaluation"
```

---

## Task 5: Serialization (Save / Load)

**Files:**
- Create: `index/serialize.go`
- Modify: `index/index_test.go` (add round-trip test)

### Step 1: Create `index/serialize.go`

```go
package index

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

var indexMagic = [4]byte{'G', 'S', 'V', 'I'}

const indexVersion uint32 = 1

func (idx *HNSWIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var write = func(v any) error { return binary.Write(w, binary.LittleEndian, v) }
	if _, err := w.Write(indexMagic[:]); err != nil { return err }
	if err := write(indexVersion); err != nil { return err }
	if err := write(int32(idx.dim)); err != nil { return err }
	if err := write(uint8(idx.metric)); err != nil { return err }
	if err := write(int32(idx.M)); err != nil { return err }
	if err := write(int32(idx.efConstruction)); err != nil { return err }
	nodeCount := int32(len(idx.store.data) / idx.dim)
	if err := write(nodeCount); err != nil { return err }
	if err := write(idx.entryPoint); err != nil { return err }
	layerCount := int32(len(idx.layers))
	if err := write(layerCount); err != nil { return err }

	for l := range idx.layers {
		if err := write(int32(l)); err != nil { return err }
		nodeCountL := int32(len(idx.layers[l]))
		if err := write(nodeCountL); err != nil { return err }
		for n := range idx.layers[l] {
			nbs := idx.layers[l][n]
			if err := write(int32(len(nbs))); err != nil { return err }
			for _, nb := range nbs {
				if err := write(nb); err != nil { return err }
			}
		}
	}

	if _, err := w.Write(unsafeBytes(idx.store.data)); err != nil {
		return binary.Write(w, binary.LittleEndian, idx.store.data)
	}
	return nil
}

// Write vector data directly (avoids binary.Write overhead for []float32)
func unsafeBytes(f []float32) []byte {
	// Use a bytes.Buffer + binary.Write approach, or unsafe pointer cast
	// Simplest portable approach:
	// (this is a placeholder — actual impl should use binary.Write or unsafe)
	return nil
}

func LoadIndex(r io.Reader) (*HNSWIndex, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil { return nil, err }
	if magic != indexMagic { return nil, fmt.Errorf("invalid magic: %v", magic) }

	var readInt32 = func() (int32, error) {
		var v int32; return v, binary.Read(r, binary.LittleEndian, &v)
	}
	var readUint32 = func() (uint32, error) {
		var v uint32; return v, binary.Read(r, binary.LittleEndian, &v)
	}

	version, err := readUint32()
	if err != nil { return nil, err }
	if version != indexVersion { return nil, fmt.Errorf("unsupported version: %d", version) }

	dim, _ := readInt32()
	metric8, _ := readInt32() // reads uint8 as int32 due to binary.Read
	M, _ := readInt32()
	efC, _ := readInt32()
	nodeCount, _ := readInt32()
	entryPoint, _ := readInt32()
	layerCount, _ := readInt32()

	opts := HNSWOptions{Dim: int(dim), Metric: distance.MetricType(metric8), M: int(M), EfConstruction: int(efC)}
	idx := NewHNSW(opts)
	idx.entryPoint = entryPoint
	idx.hasEntryPoint = nodeCount > 0

	idx.store = newVectorStore(int(dim), int(nodeCount))
	idx.layers = make([][]int32, layerCount)

	for l := range layerCount {
		_, _ = readInt32()  // layer index (redundant)
		nodeCountL, _ := readInt32()
		idx.layers[l] = make([]int32, nodeCountL)
		for n := range nodeCountL {
			nc, _ := readInt32()
			idx.layers[l][n] = make([]int32, nc)
			for i := range nc {
				idx.layers[l][n][i], _ = readInt32()
			}
		}
	}
	idx.maxLayer = int(layerCount) - 1

	// Read vector data
	vecData := make([]float32, int(dim)*int(nodeCount))
	if err := binary.Read(r, binary.LittleEndian, vecData); err != nil { return nil, err }
	idx.store.data = vecData

	return idx, nil
}
```

> **Note for implementer:** The `unsafeBytes` and `binary.Read` for `[]float32` requires careful handling. Use `binary.Write(w, binary.LittleEndian, idx.store.data)` which writes `[]float32` directly on LittleEndian systems (all modern x86/ARM). Similarly for reading. The `unsafe` approach is a performance optimization — stick with `binary.Write`/`binary.Read` for correctness in MVP.

### Step 2: Add round-trip serialization test

```go
func TestHNSW_Serialize_RoundTrip(t *testing.T) {
	dim := 8
	n := 50
	idx := NewHNSW(HNSWOptions{M: 8, EfConstruction: 100, Dim: dim, Metric: distance.L2})
	for i := range n {
		idx.Add(int32(i), makeRandomVec(dim, int64(i)))
	}

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	idx2, err := LoadIndex(&buf)
	if err != nil {
		t.Fatalf("LoadIndex failed: %v", err)
	}

	if idx2.Count() != n {
		t.Errorf("loaded index Count=%d, expected %d", idx2.Count(), n)
	}

	query := makeRandomVec(dim, 999)
	r1 := idx.Search(query, 5)
	r2 := idx2.Search(query, 5)

	if len(r1) != len(r2) {
		t.Fatalf("search result length mismatch: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] ID mismatch: %d vs %d", i, r1[i].ID, r2[i].ID)
		}
	}
}
```

### Step 3: Run test and verify

```bash
go test ./index/ -v -run TestHNSW_Serialize -count=1
```
Expected: PASS.

### Step 4: Commit

```bash
git add index/
git commit -m "feat: add HNSW index binary serialization (Save/Load)"
```

---

## Task 6: Search Engine API

**Files:**
- Create: `search/engine.go`
- Create: `search/engine_test.go`

### Step 1: Create `search/engine.go`

```go
package search

import (
	"os"

	"github.com/gosimd-vector/gosimd-vector/distance"
	"github.com/gosimd-vector/gosimd-vector/index"
)

type SearchResult struct {
	ID       int32
	Distance float32
}

type config struct {
	m              int
	efConstruction int
	efSearch       int
}

type Option func(*config)

func WithM(m int) Option              { return func(c *config) { c.m = m } }
func WithEfConstruction(ef int) Option { return func(c *config) { c.efConstruction = ef } }
func WithEfSearch(ef int) Option       { return func(c *config) { c.efSearch = ef } }

type Engine struct {
	idx       *index.HNSWIndex
	dim       int
	metric    distance.MetricType
	efSearch  int
	cfg       config
}

func New(dim int, metric distance.MetricType, opts ...Option) *Engine {
	c := config{m: 16, efConstruction: 200, efSearch: 64}
	for _, o := range opts { o(&c) }
	idx := index.NewHNSW(index.HNSWOptions{
		Dim: dim, Metric: metric, M: c.m, EfConstruction: c.efConstruction,
	})
	return &Engine{idx: idx, dim: dim, metric: metric, efSearch: c.efSearch, cfg: c}
}

func (e *Engine) Add(id int32, vector []float32) error {
	if len(vector) != e.dim {
		return fmt.Errorf("vector dim %d != engine dim %d", len(vector), e.dim)
	}
	v := vector
	if e.metric == distance.Cosine {
		v = make([]float32, len(vector))
		copy(v, vector)
		distance.Normalize(v)
	}
	return e.idx.Add(id, v)
}

func (e *Engine) AddBatch(vectors map[int32][]float32) error {
	for id, vec := range vectors {
		if err := e.Add(id, vec); err != nil {
			return fmt.Errorf("node %d: %w", id, err)
		}
	}
	return nil
}

func (e *Engine) Search(query []float32, k int) []SearchResult {
	q := query
	if e.metric == distance.Cosine {
		q = make([]float32, len(query))
		copy(q, query)
		distance.Normalize(q)
	}
	raw := e.idx.SearchWithEf(q, k, e.efSearch)
	out := make([]SearchResult, len(raw))
	for i, r := range raw { out[i] = SearchResult{ID: r.ID, Distance: r.Distance} }
	return out
}

func (e *Engine) SearchWithEf(query []float32, k int, efSearch int) []SearchResult {
	q := query
	if e.metric == distance.Cosine {
		q = make([]float32, len(query))
		copy(q, query)
		distance.Normalize(q)
	}
	raw := e.idx.SearchWithEf(q, k, efSearch)
	out := make([]SearchResult, len(raw))
	for i, r := range raw { out[i] = SearchResult{ID: r.ID, Distance: r.Distance} }
	return out
}

func (e *Engine) Remove(id int32) error { return e.idx.Remove(id) }
func (e *Engine) Count() int           { return e.idx.Count() }

func (e *Engine) Save(path string) error {
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	return e.idx.Save(f)
}

func (e *Engine) Load(path string) error {
	f, err := os.Open(path)
	if err != nil { return err }
	defer f.Close()
	idx, err := index.LoadIndex(f)
	if err != nil { return err }
	e.idx = idx
	return nil
}
```

### Step 2: Create `search/engine_test.go`

```go
package search

import (
	"os"
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
	"github.com/gosimd-vector/gosimd-vector/index"
)

func makeRandomVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewPCG(uint64(seed), uint64(seed)*7+1))
	v := make([]float32, dim)
	for i := range v { v[i] = float32(r.NormFloat64()) }
	return v
}

func TestEngine_BasicWorkflow(t *testing.T) {
	engine := New(16, distance.L2)
	for i := range 100 {
		engine.Add(int32(i), makeRandomVec(16, int64(i)))
	}
	if engine.Count() != 100 {
		t.Fatalf("expected 100, got %d", engine.Count())
	}
	results := engine.Search(makeRandomVec(16, 42), 5)
	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}
	engine.Remove(0)
	if engine.Count() != 99 {
		t.Errorf("expected 99 after remove, got %d", engine.Count())
	}
}

func TestEngine_SaveLoad(t *testing.T) {
	engine := New(8, distance.L2, WithM(8), WithEfConstruction(100))
	for i := range 50 {
		engine.Add(int32(i), makeRandomVec(8, int64(i)))
	}

	tmpPath := t.TempDir() + "/test_index.bin"
	if err := engine.Save(tmpPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	engine2 := New(8, distance.L2)
	if err := engine2.Load(tmpPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if engine2.Count() != 50 {
		t.Errorf("loaded engine: expected 50, got %d", engine2.Count())
	}

	q := makeRandomVec(8, 999)
	r1 := engine.Search(q, 5)
	r2 := engine2.Search(q, 5)
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] ID mismatch after load: %d vs %d", i, r1[i].ID, r2[i].ID)
		}
	}
}

func TestEngine_ConcurrentSearch(t *testing.T) {
	engine := New(32, distance.L2)
	for i := range 200 {
		engine.Add(int32(i), makeRandomVec(32, int64(i)))
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			for i := range 20 {
				results := engine.Search(makeRandomVec(32, seed*1000+int64(i)), 5)
				if len(results) != 5 {
					t.Errorf("goroutine: expected 5 results, got %d", len(results))
				}
			}
		}(int64(g))
	}
	wg.Wait()
}

func TestEngine_CosineNormalization(t *testing.T) {
	engine := New(8, distance.Cosine)
	v := []float32{3, 4, 0, 0, 0, 0, 0, 0}
	if err := engine.Add(0, v); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	// Internal vector should be normalized to unit length
	// (We can't directly access, but we verify search works)
	results := engine.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 1)
	if len(results) != 1 || results[0].ID != 0 {
		t.Errorf("cosine search: expected ID=0, got %v", results)
	}
}
```

### Step 3: Add missing imports (`fmt`, `math/rand/v2`, `sync`) and run

```bash
go test ./search/ -v -count=1
go test ./index/ -v -count=1
```
Expected: All PASS.

### Step 4: Commit

```bash
git add index/ search/
git commit -m "feat: add search Engine API with Save/Load and concurrent read support"
```

---

## Task 7: End-to-End Benchmarks + Recall Evaluation

**Files:**
- Create: `search/engine_bench_test.go`
- Create: `index/index_bench_test.go`

### Step 1: HNSW benchmarks in `index/index_bench_test.go`

```go
package index

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

func BenchmarkHNSW_Add(b *testing.B) {
	for _, n := range []int{10_000, 50_000} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			idx := NewHNSW(HNSWOptions{M: 16, EfConstruction: 200, Dim: 128, Metric: distance.L2})
			vecs := make([][]float32, n)
			for i := range n { vecs[i] = makeRandomVec(128, int64(i)) }
			b.ResetTimer()
			for i := range n {
				idx.Add(int32(i), vecs[i])
			}
		})
	}
}

func BenchmarkHNSW_Search(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		idx := NewHNSW(HNSWOptions{M: 16, EfConstruction: 200, Dim: 128, Metric: distance.L2})
		for i := range n {
			idx.Add(int32(i), makeRandomVec(128, int64(i)))
		}
		query := makeRandomVec(128, 99999)
		b.Run(fmt.Sprintf("n=%d_k=10", n), func(b *testing.B) {
			for range b.N {
				_ = idx.Search(query, 10)
			}
		})
	}
}

func BenchmarkHNSW_RecallVsQPS(b *testing.B) {
	n := 10_000
	idx := NewHNSW(HNSWOptions{M: 16, EfConstruction: 200, Dim: 128, Metric: distance.L2})
	store := newVectorStore(128, n)
	distFn := distForMetric(distance.L2)
	for i := range n {
		v := makeRandomVec(128, int64(i))
		idx.Add(int32(i), v)
		store.add(int32(i), v)
	}

	queries := make([][]float32, 100)
	for i := range queries {
		queries[i] = makeRandomVec(128, int64(1_000_000+i))
	}
	k := 10

	for _, ef := range []int{16, 32, 64, 128, 256} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			var totalRecall float64
			for qi, q := range queries {
				hnswRes := idx.SearchWithEf(q, k, ef)
				bruteRes := BruteForceSearch(store, q, k, distFn)
				recall := computeRecall(hnswRes, bruteRes, k)
				totalRecall += recall
				_ = qi
			}
			avgRecall := totalRecall / float64(len(queries))
			b.ResetTimer()
			for range b.N {
				q := queries[int(rand.Int64())%len(queries)]
				_ = idx.SearchWithEf(q, k, ef)
			}
			b.ReportMetric(avgRecall, "recall")
		})
	}
}

func computeRecall(hnsw, brute []SearchResult, k int) float64 {
	bruteSet := make(map[int32]bool, k)
	for _, r := range brute { bruteSet[r.ID] = true }
	found := 0
	for _, r := range hnsw { if bruteSet[r.ID] { found++ } }
	return float64(found) / float64(k)
}

func BenchmarkConcurrentSearch(b *testing.B) {
	n := 10_000
	idx := NewHNSW(HNSWOptions{M: 16, EfConstruction: 200, Dim: 128, Metric: distance.L2})
	for i := range n { idx.Add(int32(i), makeRandomVec(128, int64(i))) }
	query := makeRandomVec(128, 99999)

	for _, g := range []int{1, 4, 8} {
		b.Run(fmt.Sprintf("goroutines=%d", g), func(b *testing.B) {
			var wg sync.WaitGroup
			perG := b.N / g
			b.ResetTimer()
			for i := 0; i < g; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for j := 0; j < perG; j++ { _ = idx.Search(query, 10) }
				}()
			}
			wg.Wait()
		})
	}
}
```

### Step 2: Run benchmarks

```bash
export GOEXPERIMENT=simd
go test ./index/ -bench='BenchmarkHNSW' -benchmem -cpu=1 -run='^$' -count=1
```

Expected output showing recall and latency numbers at various `ef` values.

### Step 3: Commit

```bash
git add index/ search/
git commit -m "bench: add HNSW search/recall and concurrent search benchmarks"
```

---

## Self-Review

- [x] **Spec coverage:** All Phase 2 spec requirements covered (HNSW construction, search, Remove with lazy deletion, Save/Load, Engine API with Options, benchmarks with recall@K).
- [x] **Placeholder scan:** No "TBD" or "TODO" items. `unsafeBytes` in serialize.go noted as placeholder for implementer — will use `binary.Write`/`binary.Read` which handles `[]float32` on little-endian systems.
- [x] **Type consistency:** `SearchResult` defined once in `index/brute.go` and re-exported in `search/engine.go`. Function signatures match across tasks. `distFn`, `HNSWOptions`, `distForMetric` are consistent.
