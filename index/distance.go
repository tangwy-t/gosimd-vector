package index

import "github.com/tangwy-t/gosimd-vector/distance"

// distFn is the signature for internal distance functions (lower = closer).
type distFn func(a, b []float32) float32

// selectDistFn returns the appropriate distance function for the given metric type.
// For L2 metric, uses L2Squared (avoids sqrt).
// For Cosine metric, assumes vectors are pre-normalized; uses 1 - dot product.
// For InnerProduct metric, uses negative dot product.
func selectDistFn(metric distance.MetricType) distFn {
	switch metric {
	case distance.L2:
		return distance.L2Squared
	case distance.Cosine:
		return func(a, b []float32) float32 {
			return 1 - distance.DotProduct(a, b)
		}
	case distance.InnerProduct:
		return func(a, b []float32) float32 {
			return -distance.DotProduct(a, b)
		}
	default:
		return distance.L2Squared
	}
}
