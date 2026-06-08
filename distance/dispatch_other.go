//go:build !(amd64 || arm64)

package distance

func init() {
	dotProductImpl = dotProductScalar
	l2SquaredImpl = l2SquaredScalar
	l2Impl = l2Scalar
	cosineImpl = cosineScalar
	normImpl = normScalar
}
