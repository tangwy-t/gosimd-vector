package index

import "errors"

// vectorStore holds a contiguous array of all vectors for cache-friendly access.
// Vector i occupies data[i*dim:(i+1)*dim].
type vectorStore struct {
	data    []float32
	dim     int
	deleted []bool
	ids     []int32 // external ID -> internal index mapping
	count   int
}

func newVectorStore(dim int, capacity int) *vectorStore {
	return &vectorStore{
		data:    make([]float32, 0, dim*capacity),
		dim:     dim,
		deleted: make([]bool, 0, capacity),
		ids:     make([]int32, 0, capacity),
		count:   0,
	}
}

// add appends a vector and returns its internal index
func (s *vectorStore) add(vec []float32) (int32, error) {
	if len(vec) != s.dim {
		return -1, errors.New("vector dimension mismatch")
	}
	idx := int32(s.count)
	s.data = append(s.data, vec...)
	s.deleted = append(s.deleted, false)
	s.ids = append(s.ids, idx)
	s.count++
	return idx, nil
}

// get returns the vector at the given internal index
func (s *vectorStore) get(idx int32) []float32 {
	start := int(idx) * s.dim
	return s.data[start : start+s.dim]
}

// remove marks a vector as deleted
func (s *vectorStore) remove(idx int32) {
	if int(idx) < len(s.deleted) {
		s.deleted[idx] = true
	}
}

// isDeleted checks if a vector has been removed
func (s *vectorStore) isDeleted(idx int32) bool {
	if int(idx) >= len(s.deleted) {
		return true
	}
	return s.deleted[idx]
}

// size returns the number of vectors (including deleted)
func (s *vectorStore) size() int32 {
	return int32(s.count)
}
