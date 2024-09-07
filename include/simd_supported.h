#pragma once


namespace simd_supported {
    constexpr int ix86_fp{
#ifdef _M_IX86_FP
            _M_IX86_FP
#else
            -1
#endif
    };
    constexpr bool mmx{
#ifdef __MMX__
            true
#endif
    };

    constexpr bool sse{
#ifdef __SSE__
            true
#endif
    };

    constexpr bool sse2{
#ifdef __SSE2__
            true
#endif
    };

    constexpr bool sse3{
#ifdef __SSE3__
            true
#endif
    };

    constexpr bool ssse3{
#ifdef __SSSE3__
            true
#endif
    };

    constexpr bool sse4_1{
#ifdef __SSE4_1__
            true
#endif
    };

    constexpr bool sse4_2{
#ifdef __SSE4_2__
            true
#endif
    };

    // include sse family
    constexpr bool avx{
#ifdef __AVX__
            true
#endif
    };

    // include sse family + avx
    constexpr bool avx2{
#ifdef __AVX2__
            true
#endif
    };

    // include sse family + avx family
    constexpr bool avx512f{
#ifdef __AVX512F__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512cd{
#ifdef __AVX512CD__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512vl{
#ifdef __AVX512VL__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512bw{
#ifdef __AVX512BW__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512dq{
#ifdef __AVX512DQ__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512ifma{
#ifdef __AVX512IFMA__
            true
#endif
    };

    // include avx512bw + avx512f + sse family + avx family
    constexpr bool avx512vbmi{
#ifdef __AVX512VBMI__
            true
#endif
    };

    // include avx512bw + avx512f + sse family + avx family
    constexpr bool avx512vbmi2{
#ifdef __AVX512VBMI2__
            true
#endif
    };

    // include avx512bw + avx512f + sse family + avx family
    constexpr bool avx512bf16{
#ifdef __AVX512BF16__
            true
#endif
    };

    // include avx512dq + avx512bw + avx512vl + avx512f + sse family + avx family
    constexpr bool avx512fp16{
#ifdef __AVX512FP16__
            true
#endif
    };

    // include avx512bw + avx512f + sse family + avx family
    constexpr bool avx512bitalg{
#ifdef __AVX512BITALG__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512vpopcntdq{
#ifdef __AVX512VPOPCNTDQ__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512vp2intersect{
#ifdef __AVX512VP2INTERSECT__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512vnni{
#ifdef __AVX512VNNI__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512pf{
#ifdef __AVX512PF__
            true
#endif
    };

    // include avx512f + sse family + avx family
    constexpr bool avx512er{
#ifdef __AVX512ER__
            true
#endif
    };

    constexpr bool avx5124fmaps{
#ifdef __AVX5124FMAPS__
            true
#endif
    };

    constexpr bool avx5124vnniw{
#ifdef __AVX5124VNNIW__
            true
#endif
    };
}  // namespace simd_supported
