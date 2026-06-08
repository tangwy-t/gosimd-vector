package distance

type MetricType int

const (
	L2           MetricType = iota
	Cosine
	InnerProduct
)

type distanceFn func(a, b []float32) float32
type normFn func(v []float32)

var (
	dotProductImpl distanceFn
	l2SquaredImpl  distanceFn
	l2Impl         distanceFn
	cosineImpl     distanceFn
	normImpl       normFn
)

func DotProduct(a, b []float32) float32      { return dotProductImpl(a, b) }
func L2Squared(a, b []float32) float32        { return l2SquaredImpl(a, b) }
func L2Distance(a, b []float32) float32       { return l2Impl(a, b) }
func CosineSimilarity(a, b []float32) float32 { return cosineImpl(a, b) }
func Normalize(v []float32)                   { normImpl(v) }
