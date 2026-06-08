package distance

import "math"

func dotProductScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func l2SquaredScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func l2Scalar(a, b []float32) float32 {
	return float32(math.Sqrt(float64(l2SquaredScalar(a, b))))
}

func cosineScalar(a, b []float32) float32 {
	dot := dotProductScalar(a, b)
	na := dotProductScalar(a, a)
	nb := dotProductScalar(b, b)
	denom := float32(math.Sqrt(float64(na))) * float32(math.Sqrt(float64(nb)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func normScalar(v []float32) {
	sq := dotProductScalar(v, v)
	if sq == 0 {
		for i := range v {
			v[i] = 0
		}
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sq)))
	for i := range v {
		v[i] *= inv
	}
}
