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
		cosineImpl = cosineFusedAVX512
		normImpl = normAVX512
	case hasAVX2:
		dotProductImpl = dotProductAVX2
		l2SquaredImpl = l2SquaredAVX2
		l2Impl = l2AVX2
		cosineImpl = cosineFusedAVX2
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

func cosineFusedAVX2(a, b []float32) float32 {
	n := len(a)
	sumDot := archsimd.BroadcastFloat32x8(0)
	sumA := archsimd.BroadcastFloat32x8(0)
	sumB := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		sumDot = va.MulAdd(vb, sumDot)
		sumA = va.MulAdd(va, sumA)
		sumB = vb.MulAdd(vb, sumB)
	}

	loD, hiD := sumDot.GetLo(), sumDot.GetHi()
	loA, hiA := sumA.GetLo(), sumA.GetHi()
	loB, hiB := sumB.GetLo(), sumB.GetHi()

	vd := loD.Add(hiD)
	vaR := loA.Add(hiA)
	vbR := loB.Add(hiB)
	vd = vd.AddPairs(vd)
	vd = vd.AddPairs(vd)
	vaR = vaR.AddPairs(vaR)
	vaR = vaR.AddPairs(vaR)
	vbR = vbR.AddPairs(vbR)
	vbR = vbR.AddPairs(vbR)

	resultDot := vd.GetElem(0)
	resultA := vaR.GetElem(0)
	resultB := vbR.GetElem(0)
	for ; i < n; i++ {
		resultDot += a[i] * b[i]
		resultA += a[i] * a[i]
		resultB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(resultA))) * float32(math.Sqrt(float64(resultB)))
	if denom == 0 {
		return 0
	}
	return resultDot / denom
}

func cosineFusedAVX512(a, b []float32) float32 {
	n := len(a)
	sumDot := archsimd.BroadcastFloat32x16(0)
	sumA := archsimd.BroadcastFloat32x16(0)
	sumB := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadFloat32x16Slice(a[i:])
		vb := archsimd.LoadFloat32x16Slice(b[i:])
		sumDot = va.MulAdd(vb, sumDot)
		sumA = va.MulAdd(va, sumA)
		sumB = vb.MulAdd(vb, sumB)
	}

	lo16D, hi16D := sumDot.GetLo(), sumDot.GetHi()
	lo16A, hi16A := sumA.GetLo(), sumA.GetHi()
	lo16B, hi16B := sumB.GetLo(), sumB.GetHi()

	s8D := lo16D.Add(hi16D)
	s8A := lo16A.Add(hi16A)
	s8B := lo16B.Add(hi16B)

	lo8D, hi8D := s8D.GetLo(), s8D.GetHi()
	lo8A, hi8A := s8A.GetLo(), s8A.GetHi()
	lo8B, hi8B := s8B.GetLo(), s8B.GetHi()

	vd := lo8D.Add(hi8D)
	vaR := lo8A.Add(hi8A)
	vbR := lo8B.Add(hi8B)
	vd = vd.AddPairs(vd)
	vd = vd.AddPairs(vd)
	vaR = vaR.AddPairs(vaR)
	vaR = vaR.AddPairs(vaR)
	vbR = vbR.AddPairs(vbR)
	vbR = vbR.AddPairs(vbR)

	resultDot := vd.GetElem(0)
	resultA := vaR.GetElem(0)
	resultB := vbR.GetElem(0)
	for ; i < n; i++ {
		resultDot += a[i] * b[i]
		resultA += a[i] * a[i]
		resultB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(resultA))) * float32(math.Sqrt(float64(resultB)))
	if denom == 0 {
		return 0
	}
	return resultDot / denom
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
