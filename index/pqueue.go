package index

import "container/heap"

// candidate represents a vector ID with its distance from a query
type candidate struct {
	id   int32
	dist float32
}

// candidateList implements heap.Interface for min-heap (closest first)
type candidateList []candidate

func (cl candidateList) Len() int            { return len(cl) }
func (cl candidateList) Swap(i, j int)       { cl[i], cl[j] = cl[j], cl[i] }
func (cl candidateList) Less(i, j int) bool  { return cl[i].dist < cl[j].dist }
func (cl *candidateList) Push(x any)         { *cl = append(*cl, x.(candidate)) }
func (cl *candidateList) Pop() any           { old := *cl; n := len(old); x := old[n-1]; *cl = old[:n-1]; return x }

// maxCandidateList implements heap.Interface for max-heap (farthest first)
type maxCandidateList []candidate

func (cl maxCandidateList) Len() int            { return len(cl) }
func (cl maxCandidateList) Swap(i, j int)       { cl[i], cl[j] = cl[j], cl[i] }
func (cl maxCandidateList) Less(i, j int) bool  { return cl[i].dist > cl[j].dist }
func (cl *maxCandidateList) Push(x any)         { *cl = append(*cl, x.(candidate)) }
func (cl *maxCandidateList) Pop() any           { old := *cl; n := len(old); x := old[n-1]; *cl = old[:n-1]; return x }

// MinHeap wrapper for closest-first priority queue
type MinHeap struct {
	data candidateList
}

func newMinHeap() *MinHeap {
	h := &MinHeap{data: make(candidateList, 0)}
	heap.Init(&h.data)
	return h
}

func (h *MinHeap) Push(c candidate) {
	heap.Push(&h.data, c)
}

func (h *MinHeap) Pop() candidate {
	return heap.Pop(&h.data).(candidate)
}

func (h *MinHeap) Len() int {
	return h.data.Len()
}

func (h *MinHeap) Peek() candidate {
	return h.data[0]
}

// MaxHeap wrapper for farthest-first priority queue
type MaxHeap struct {
	data maxCandidateList
}

func newMaxHeap() *MaxHeap {
	h := &MaxHeap{data: make(maxCandidateList, 0)}
	heap.Init(&h.data)
	return h
}

func (h *MaxHeap) Push(c candidate) {
	heap.Push(&h.data, c)
}

func (h *MaxHeap) Pop() candidate {
	return heap.Pop(&h.data).(candidate)
}

func (h *MaxHeap) Len() int {
	return h.data.Len()
}

func (h *MaxHeap) Peek() candidate {
	return h.data[0]
}
