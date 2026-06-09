package index

import (
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

type Config struct {
	Dim            int
	Metric         distance.MetricType
	M              int
	EfConstruction int
}

type HNSW struct {
	mu         sync.RWMutex
	cfg        Config
	dist       distFn
	store      *vectorStore
	graph      []map[int32][]int32
	entryPoint int32
	entryLevel int
	levelMult  float64
	rng        *rand.Rand
	count      int
	extToInt   map[int32]int32
	intToExt   map[int32]int32
}

func NewHNSW(cfg Config) *HNSW {
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	return &HNSW{
		cfg:        cfg,
		dist:       selectDistFn(cfg.Metric),
		store:      newVectorStore(cfg.Dim, 1024),
		graph:      []map[int32][]int32{make(map[int32][]int32)},
		entryPoint: -1,
		entryLevel: -1,
		levelMult:  1.0 / math.Log(float64(cfg.M)),
		rng:        rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
		extToInt:   make(map[int32]int32),
		intToExt:   make(map[int32]int32),
	}
}

func (h *HNSW) randomLevel() int {
	return int(-math.Log(h.rng.Float64()) * h.levelMult)
}

func (h *HNSW) Count() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.count
}

func (h *HNSW) singleNearest(query []float32, start int32, layer int) int32 {
	cur := start
	curDist := h.dist(query, h.store.get(cur))
	visited := make(map[int32]struct{})
	visited[cur] = struct{}{}
	for {
		improved := false
		if layer < len(h.graph) {
			for _, nb := range h.graph[layer][cur] {
				if h.store.isDeleted(nb) {
					continue
				}
				if _, ok := visited[nb]; ok {
					continue
				}
				visited[nb] = struct{}{}
				d := h.dist(query, h.store.get(nb))
				if d < curDist {
					cur = nb
					curDist = d
					improved = true
				}
			}
		}
		if !improved {
			break
		}
	}
	return cur
}

func (h *HNSW) beamSearch(query []float32, entry int32, layer int, ef int) []candidate {
	if h.store.isDeleted(entry) {
		return nil
	}
	visited := make(map[int32]struct{})
	visited[entry] = struct{}{}

	c := newMinHeap()
	w := newMaxHeap()

	d := h.dist(query, h.store.get(entry))
	c.Push(candidate{entry, d})
	w.Push(candidate{entry, d})

	for c.Len() > 0 {
		nearest := c.Peek()
		farthest := w.Peek()
		if nearest.dist > farthest.dist {
			break
		}
		c.Pop()

		if layer < len(h.graph) {
			for _, nb := range h.graph[layer][nearest.id] {
				if _, ok := visited[nb]; ok {
					continue
				}
				visited[nb] = struct{}{}
				if h.store.isDeleted(nb) {
					continue
				}
				nbDist := h.dist(query, h.store.get(nb))
				farthest = w.Peek()
				if nbDist < farthest.dist || w.Len() < ef {
					c.Push(candidate{nb, nbDist})
					w.Push(candidate{nb, nbDist})
					if w.Len() > ef {
						w.Pop()
					}
				}
			}
		}
	}
	return w.toArray()
}

func (h *HNSW) connect(newNode int32, neighbors []candidate, layer int) {
	maxConns := h.cfg.M
	if layer > 0 {
		maxConns = h.cfg.M / 2
	}
	if maxConns <= 0 {
		maxConns = 1
	}

	if layer >= len(h.graph) {
		return
	}

	conns := neighbors
	if len(conns) > maxConns {
		conns = conns[:maxConns]
	}

	newNodeNeighbors := make([]int32, len(conns))
	for i, c := range conns {
		newNodeNeighbors[i] = c.id

		existing := h.graph[layer][c.id]
		existing = append(existing, newNode)
		if len(existing) > maxConns {
			vec := h.store.get(c.id)
			scored := make([]candidate, len(existing))
			for j, nb := range existing {
				scored[j] = candidate{nb, h.dist(vec, h.store.get(nb))}
			}
			sort.Slice(scored, func(i, j int) bool { return scored[i].dist < scored[j].dist })
			pruned := make([]int32, maxConns)
			for j := range maxConns {
				pruned[j] = scored[j].id
			}
			existing = pruned
		}
		h.graph[layer][c.id] = existing
	}
	h.graph[layer][newNode] = newNodeNeighbors
}

func (h *HNSW) Add(id int32, vec []float32) {
	level := h.randomLevel()

	h.mu.Lock()
	defer h.mu.Unlock()

	idx, err := h.store.add(vec)
	if err != nil {
		return
	}

	h.extToInt[id] = idx
	h.intToExt[idx] = id
	h.count++

	for level >= len(h.graph) {
		h.graph = append(h.graph, make(map[int32][]int32))
	}

	if h.entryPoint == -1 {
		h.entryPoint = idx
		h.entryLevel = level
		return
	}

	cur := h.entryPoint
	for lc := h.entryLevel; lc > level; lc-- {
		cur = h.singleNearest(vec, cur, lc)
	}

	startLayer := level
	if h.entryLevel < startLayer {
		startLayer = h.entryLevel
	}
	for lc := startLayer; lc >= 0; lc-- {
		candidates := h.beamSearch(vec, cur, lc, h.cfg.EfConstruction)
		h.connect(idx, candidates, lc)
		if len(candidates) > 0 {
			cur = candidates[0].id
		}
	}

	if level > h.entryLevel {
		h.entryPoint = idx
		h.entryLevel = level
	}
}

func (h *HNSW) Remove(id int32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	idx, ok := h.extToInt[id]
	if !ok {
		return
	}
	if h.store.isDeleted(idx) {
		return
	}
	h.store.remove(idx)
	h.count--
}

func (h *HNSW) Search(query []float32, topK int) []SearchResult {
	return h.SearchWithEf(query, topK, h.cfg.EfConstruction)
}

func (h *HNSW) SearchWithEf(query []float32, topK int, ef int) []SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		return nil
	}
	if ef < topK {
		ef = topK
	}

	cur := h.entryPoint
	for lc := h.entryLevel; lc > 0; lc-- {
		cur = h.singleNearest(query, cur, lc)
	}

	candidates := h.beamSearch(query, cur, 0, ef)

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	results := make([]SearchResult, 0, topK)
	for _, c := range candidates {
		if h.store.isDeleted(c.id) {
			continue
		}
		extID := h.intToExt[c.id]
		results = append(results, SearchResult{ID: extID, Distance: c.dist})
		if len(results) >= topK {
			break
		}
	}
	return results
}
