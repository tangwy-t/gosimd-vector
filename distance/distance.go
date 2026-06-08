package distance

type MetricType int

const (
	L2           MetricType = iota
	Cosine
	InnerProduct
)
