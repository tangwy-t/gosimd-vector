package index

import "sort"

// SearchResult holds a single nearest-neighbor result
type SearchResult struct {
	ID       int32
	Distance float32
}

// BruteForceSearch performs exhaustive KNN search over all non-deleted vectors.
// Used as ground truth for HNSW recall evaluation.
func BruteForceSearch(store *vectorStore, query []float32, k int, dist distFn) []SearchResult {
	n := store.size()
	results := make([]SearchResult, 0, n)
	for id := int32(0); id < n; id++ {
		if store.isDeleted(id) {
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
