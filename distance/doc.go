// Package distance provides SIMD-accelerated vector distance computation.
//
// Supported metrics: DotProduct (inner product), CosineSimilarity,
// L2Distance (Euclidean), and L2Squared (squared Euclidean). Normalize
// converts vectors to unit length in-place.
//
// On amd64, AVX2 and AVX-512 are selected automatically at startup based
// on CPU capability. On arm64, NEON is always used. On other architectures,
// a scalar fallback is used.
//
// All functions require len(a) == len(b). Passing mismatched lengths
// results in undefined behavior; callers must ensure equal lengths.
//
// Requires Go 1.26+ with GOEXPERIMENT=simd for simd/archsimd support.
package distance
