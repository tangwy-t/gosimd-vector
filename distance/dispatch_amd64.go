//go:build amd64

package distance

import (
	"math"
	"simd/archsimd"

	"golang.org/x/sys/cpu"
)

var (
	hasAVX2   bool
	hasAVX512 bool
)

func init() {
	hasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasAVX512 = cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ

	switch {
	case hasAVX512:
		dotProductImpl = dotProductAVX512
		l2SquaredImpl = l2SquaredAVX512
		l2Impl = l2AVX512
		cosineImpl = cosineAVX512
		normImpl = normAVX512
	case hasAVX2:
		dotProductImpl = dotProductAVX2
		l2SquaredImpl = l2SquaredAVX2
		l2Impl = l2AVX2
		cosineImpl = cosineAVX2
		normImpl = normAVX2
	default:
		dotProductImpl = dotProductScalar
		l2SquaredImpl = l2SquaredScalar
		l2Impl = l2Scalar
		cosineImpl = cosineScalar
		normImpl = normScalar
	}
}

func dotProductAVX2(a, b []float32) float32 {
	n := len(a)
	sum := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		sum = va.MulAdd(vb, sum)
	}
	lo := sum.GetLo()
	hi := sum.GetHi()
	v4 := lo.Add(hi)
	v := v4.AddPairs(v4)
	v = v.AddPairs(v)
	result := v.GetElem(0)
	for ; i < n; i++ {
		result += a[i] * b[i]
	}
	return result
}

func dotProductAVX512(a, b []float32) float32 {
	n := len(a)
	sum := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadFloat32x16Slice(a[i:])
		vb := archsimd.LoadFloat32x16Slice(b[i:])
		sum = va.MulAdd(vb, sum)
	}
	lo16 := sum.GetLo()
	hi16 := sum.GetHi()
	s8 := lo16.Add(hi16)
	lo8 := s8.GetLo()
	hi8 := s8.GetHi()
	v4 := lo8.Add(hi8)
	v := v4.AddPairs(v4)
	v = v.AddPairs(v)
	result := v.GetElem(0)
	for ; i < n; i++ {
		result += a[i] * b[i]
	}
	return result
}

func l2SquaredAVX2(a, b []float32) float32 {
	n := len(a)
	sum := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		diff := va.Sub(vb)
		sum = diff.MulAdd(diff, sum)
	}
	lo := sum.GetLo()
	hi := sum.GetHi()
	v4 := lo.Add(hi)
	v := v4.AddPairs(v4)
	v = v.AddPairs(v)
	result := v.GetElem(0)
	for ; i < n; i++ {
		d := a[i] - b[i]
		result += d * d
	}
	return result
}

func l2SquaredAVX512(a, b []float32) float32 {
	n := len(a)
	sum := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadFloat32x16Slice(a[i:])
		vb := archsimd.LoadFloat32x16Slice(b[i:])
		diff := va.Sub(vb)
		sum = diff.MulAdd(diff, sum)
	}
	lo16 := sum.GetLo()
	hi16 := sum.GetHi()
	s8 := lo16.Add(hi16)
	lo8 := s8.GetLo()
	hi8 := s8.GetHi()
	v4 := lo8.Add(hi8)
	v := v4.AddPairs(v4)
	v = v.AddPairs(v)
	result := v.GetElem(0)
	for ; i < n; i++ {
		d := a[i] - b[i]
		result += d * d
	}
	return result
}

func l2AVX2(a, b []float32) float32 {
	return float32(math.Sqrt(float64(l2SquaredAVX2(a, b))))
}

func l2AVX512(a, b []float32) float32 {
	return float32(math.Sqrt(float64(l2SquaredAVX512(a, b))))
}

func cosineAVX2(a, b []float32) float32 {
	dot := dotProductAVX2(a, b)
	normA := dotProductAVX2(a, a)
	normB := dotProductAVX2(b, b)
	denom := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func cosineAVX512(a, b []float32) float32 {
	dot := dotProductAVX512(a, b)
	normA := dotProductAVX512(a, a)
	normB := dotProductAVX512(b, b)
	denom := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func normAVX2(v []float32) {
	sq := dotProductAVX2(v, v)
	if sq == 0 {
		for i := range v {
			v[i] = 0
		}
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sq)))
	broadcastInv := archsimd.BroadcastFloat32x8(inv)
	n := len(v)
	i := 0
	for ; i+8 <= n; i += 8 {
		vi := archsimd.LoadFloat32x8Slice(v[i:])
		vi.Mul(broadcastInv).StoreSlice(v[i:])
	}
	for ; i < n; i++ {
		v[i] *= inv
	}
}

func normAVX512(v []float32) {
	sq := dotProductAVX512(v, v)
	if sq == 0 {
		for i := range v {
			v[i] = 0
		}
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sq)))
	broadcastInv := archsimd.BroadcastFloat32x16(inv)
	n := len(v)
	i := 0
	for ; i+16 <= n; i += 16 {
		vi := archsimd.LoadFloat32x16Slice(v[i:])
		vi.Mul(broadcastInv).StoreSlice(v[i:])
	}
	for ; i < n; i++ {
		v[i] *= inv
	}
}
