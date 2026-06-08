//go:build amd64

package distance

import "golang.org/x/sys/cpu"

func init() {
	_ = cpu.X86
	dotProductImpl = dotProductScalar
	l2SquaredImpl = l2SquaredScalar
	l2Impl = l2Scalar
	cosineImpl = cosineScalar
	normImpl = normScalar
}
