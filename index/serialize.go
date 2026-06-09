package index

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"

	"github.com/gosimd-vector/gosimd-vector/distance"
)

var magic = [4]byte{'G', 'S', 'V', '1'}

func writeI32(w io.Writer, v int32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(v))
	_, err := w.Write(buf[:])
	return err
}

func readI32(r io.Reader) (int32, error) {
	var buf [4]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return int32(binary.LittleEndian.Uint32(buf[:])), nil
}

func (h *HNSW) Save(w io.Writer) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeI32(w, 1); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.cfg.Dim)); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.cfg.Metric)); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.cfg.M)); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.cfg.EfConstruction)); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.count)); err != nil {
		return err
	}
	if err := writeI32(w, h.entryPoint); err != nil {
		return err
	}
	if err := writeI32(w, int32(h.entryLevel)); err != nil {
		return err
	}
	numLayers := int32(len(h.graph))
	if err := writeI32(w, numLayers); err != nil {
		return err
	}

	for _, layer := range h.graph {
		if err := writeI32(w, int32(len(layer))); err != nil {
			return err
		}
		for nodeIdx, neighbors := range layer {
			if err := writeI32(w, nodeIdx); err != nil {
				return err
			}
			if err := writeI32(w, int32(len(neighbors))); err != nil {
				return err
			}
			for _, nb := range neighbors {
				if err := writeI32(w, nb); err != nil {
					return err
				}
			}
		}
	}

	storeSize := h.store.size()
	if err := writeI32(w, storeSize); err != nil {
		return err
	}

	numFloats := storeSize * int32(h.cfg.Dim)
	if numFloats > 0 {
		if _, err := w.Write(float32sToBytes(h.store.data[:numFloats])); err != nil {
			return err
		}
	}

	deletedBytes := make([]byte, storeSize)
	for i := int32(0); i < storeSize; i++ {
		if h.store.isDeleted(i) {
			deletedBytes[i] = 1
		}
	}
	if storeSize > 0 {
		if _, err := w.Write(deletedBytes); err != nil {
			return err
		}
	}

	for i := int32(0); i < storeSize; i++ {
		if err := writeI32(w, i); err != nil {
			return err
		}
		extID, ok := h.intToExt[i]
		if !ok {
			return fmt.Errorf("missing external ID for internal index %d", i)
		}
		if err := writeI32(w, extID); err != nil {
			return err
		}
	}

	return nil
}

func Load(r io.Reader) (*HNSW, error) {
	var m [4]byte
	if _, err := io.ReadFull(r, m[:]); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if m != magic {
		return nil, fmt.Errorf("invalid magic: %v", m)
	}

	version, err := readI32(r)
	if err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	dim, err := readI32(r)
	if err != nil {
		return nil, err
	}
	metricInt, err := readI32(r)
	if err != nil {
		return nil, err
	}
	mVal, err := readI32(r)
	if err != nil {
		return nil, err
	}
	efConstr, err := readI32(r)
	if err != nil {
		return nil, err
	}
	activeCount, err := readI32(r)
	if err != nil {
		return nil, err
	}
	entryPoint, err := readI32(r)
	if err != nil {
		return nil, err
	}
	entryLevel, err := readI32(r)
	if err != nil {
		return nil, err
	}
	numLayers, err := readI32(r)
	if err != nil {
		return nil, err
	}

	metric := distance.MetricType(metricInt)
	cfg := Config{
		Dim:            int(dim),
		Metric:         metric,
		M:              int(mVal),
		EfConstruction: int(efConstr),
		EfSearch:       50,
	}

	graph := make([]map[int32][]int32, numLayers)
	for l := int32(0); l < numLayers; l++ {
		numNodes, err := readI32(r)
		if err != nil {
			return nil, err
		}
		layer := make(map[int32][]int32, numNodes)
		for i := int32(0); i < numNodes; i++ {
			nodeIdx, err := readI32(r)
			if err != nil {
				return nil, err
			}
			numNeighbors, err := readI32(r)
			if err != nil {
				return nil, err
			}
			neighbors := make([]int32, numNeighbors)
			for j := int32(0); j < numNeighbors; j++ {
				nb, err := readI32(r)
				if err != nil {
					return nil, err
				}
				neighbors[j] = nb
			}
			layer[nodeIdx] = neighbors
		}
		graph[l] = layer
	}

	storeSize, err := readI32(r)
	if err != nil {
		return nil, err
	}

	storeData := make([]float32, 0)
	numFloats := int(storeSize) * int(dim)
	if numFloats > 0 {
		dataBytes := make([]byte, numFloats*4)
		if _, err := io.ReadFull(r, dataBytes); err != nil {
			return nil, fmt.Errorf("reading vector data: %w", err)
		}
		storeData = bytesToFloat32s(dataBytes)
	}

	deletedFlags := make([]bool, storeSize)
	if storeSize > 0 {
		deletedBytes := make([]byte, storeSize)
		if _, err := io.ReadFull(r, deletedBytes); err != nil {
			return nil, err
		}
		for i := int32(0); i < storeSize; i++ {
			deletedFlags[i] = deletedBytes[i] == 1
		}
	}

	extToInt := make(map[int32]int32, storeSize)
	intToExt := make(map[int32]int32, storeSize)
	for i := int32(0); i < storeSize; i++ {
		intIdx, err := readI32(r)
		if err != nil {
			return nil, err
		}
		extID, err := readI32(r)
		if err != nil {
			return nil, err
		}
		extToInt[extID] = intIdx
		intToExt[intIdx] = extID
	}

	ids := make([]int32, storeSize)
	for i := int32(0); i < storeSize; i++ {
		ids[i] = i
	}

	store := &vectorStore{
		data:    storeData,
		dim:     int(dim),
		deleted: deletedFlags,
		ids:     ids,
		count:   int(storeSize),
	}

	h := &HNSW{
		cfg:        cfg,
		dist:       selectDistFn(metric),
		store:      store,
		graph:      graph,
		entryPoint: entryPoint,
		entryLevel: int(entryLevel),
		levelMult:  1.0 / math.Log(float64(cfg.M)),
		rng:        rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
		count:      int(activeCount),
		extToInt:   extToInt,
		intToExt:   intToExt,
	}

	return h, nil
}

func float32sToBytes(floats []float32) []byte {
	b := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func bytesToFloat32s(b []byte) []float32 {
	n := len(b) / 4
	floats := make([]float32, n)
	for i := 0; i < n; i++ {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return floats
}
