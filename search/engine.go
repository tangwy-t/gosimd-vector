package search

import (
	"fmt"
	"io"

	"github.com/tangwy-t/gosimd-vector/distance"
	"github.com/tangwy-t/gosimd-vector/index"
)

type SearchResult struct {
	ID       int32
	Distance float32
}

type Option func(*engineConfig)

func WithM(m int) Option { return func(c *engineConfig) { c.m = m } }
func WithEfConstruction(ef int) Option { return func(c *engineConfig) { c.efConstruction = ef } }
func WithEfSearch(ef int) Option { return func(c *engineConfig) { c.efSearch = ef } }

type engineConfig struct {
	m              int
	efConstruction int
	efSearch       int
}

type Engine struct {
	hnsw   *index.HNSW
	cfg    engineConfig
	dim    int
	metric distance.MetricType
}

func New(dim int, metric distance.MetricType, opts ...Option) *Engine {
	cfg := engineConfig{
		m:              16,
		efConstruction: 200,
		efSearch:       64,
	}
	for _, o := range opts {
		o(&cfg)
	}
	h := index.NewHNSW(index.Config{
		Dim:            dim,
		Metric:         metric,
		M:              cfg.m,
		EfConstruction: cfg.efConstruction,
	})
	return &Engine{
		hnsw:   h,
		cfg:    cfg,
		dim:    dim,
		metric: metric,
	}
}

func (e *Engine) Add(id int32, vec []float32) error {
	if len(vec) != e.dim {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", e.dim, len(vec))
	}
	if e.metric == distance.Cosine {
		v := make([]float32, len(vec))
		copy(v, vec)
		distance.Normalize(v)
		e.hnsw.Add(id, v)
	} else {
		e.hnsw.Add(id, vec)
	}
	return nil
}

func (e *Engine) AddBatch(vectors map[int32][]float32) error {
	for id, vec := range vectors {
		if err := e.Add(id, vec); err != nil {
			return err
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
	internal := e.hnsw.SearchWithEf(q, k, e.cfg.efSearch)
	results := make([]SearchResult, len(internal))
	for i, r := range internal {
		results[i] = SearchResult{ID: r.ID, Distance: r.Distance}
	}
	return results
}

func (e *Engine) SearchWithEf(query []float32, k int, efSearch int) []SearchResult {
	q := query
	if e.metric == distance.Cosine {
		q = make([]float32, len(query))
		copy(q, query)
		distance.Normalize(q)
	}
	internal := e.hnsw.SearchWithEf(q, k, efSearch)
	results := make([]SearchResult, len(internal))
	for i, r := range internal {
		results[i] = SearchResult{ID: r.ID, Distance: r.Distance}
	}
	return results
}

func (e *Engine) Remove(id int32) {
	e.hnsw.Remove(id)
}

func (e *Engine) Count() int {
	return e.hnsw.Count()
}

func (e *Engine) Save(w io.Writer) error {
	return e.hnsw.Save(w)
}

func (e *Engine) Load(r io.Reader) error {
	h, err := index.Load(r)
	if err != nil {
		return err
	}
	e.hnsw = h
	return nil
}
