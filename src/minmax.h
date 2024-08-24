#pragma once


#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cstddef>
#include <cstdint>

#include <concepts>
#include <utility>


namespace detail {
    template <typename T, typename Elem = std::remove_cvref_t<T>>
    concept VectorizeSafely = !std::is_volatile_v<std::remove_reference_t<T>> && (std::floating_point<Elem> && sizeof(Elem) <= sizeof(double) || std::integral<Elem> || std::is_pointer_v<Elem>);


    template <VectorizeSafely T>
    constexpr auto ToArithmeticType() noexcept {
        constexpr bool is_signed = std::is_signed_v<T>;

        if constexpr (std::floating_point<std::remove_cvref_t<T>>) {
            return std::type_identity_t<std::remove_cvref_t<T>>{};
        } else if constexpr (sizeof(T) == 1) {
            if constexpr (is_signed) {
                return std::type_identity_t<int8_t>{};
            } else {
                return std::type_identity_t<uint8_t>{};
            }
        } else if constexpr (sizeof(T) == 2) {
            if constexpr (is_signed) {
                return std::type_identity_t<int16_t>{};
            } else {
                return std::type_identity_t<uint16_t>{};
            }
        } else if constexpr (sizeof(T) == 4) {
            if constexpr (is_signed) {
                return std::type_identity_t<int32_t>{};
            } else {
                return std::type_identity_t<uint32_t>{};
            }
        } else {
            if constexpr (is_signed) {
                return std::type_identity_t<int64_t>{};
            } else {
                return std::type_identity_t<uint64_t>{};
            }
        }
    }


    template <size_t NumBits, typename T>
    struct VecTraits;
    // ===================================== Vec128 =========================================
#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
    template <>
    struct VecTraits<128, float> {
        using Type = __m128;

        static __m128 Blend(__m128 a, __m128 b, __m128 mask) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_blendv_ps(a, b, mask);
#else
            return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
#endif
        }

        static __m128 Load(const void* data) noexcept {
            return _mm_loadu_ps(static_cast<const float*>(data));
        }

        static __m128 Min(__m128 a, __m128 b) noexcept {
            return Blend(_mm_min_ps(b, a), b, _mm_cmpunord_ps(a, a));
        }

        static __m128 Max(__m128 a, __m128 b) noexcept {
            return Blend(_mm_max_ps(b, a), b, _mm_cmpunord_ps(a, a));
        }

        static float ReduceMin(__m128 a) noexcept {
            a = Min(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 3, 2)));
            a = Min(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
            return _mm_cvtss_f32(a);
        }

        static float ReduceMax(__m128 a) noexcept {
            a = Max(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 3, 2)));
            a = Max(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
            return _mm_cvtss_f32(a);
        }
    };

    template <>
    struct VecTraits<128, double> {
        using Type = __m128d;

        static __m128d Blend(__m128d a, __m128d b, __m128d mask) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_blendv_pd(a, b, mask);
#else
            return _mm_or_pd(_mm_and_pd(mask, b), _mm_andnot_pd(mask, a));
#endif
        }

        static __m128d Load(const void* data) noexcept {
            return _mm_loadu_pd(static_cast<const double*>(data));
        }

        static __m128d Min(__m128d a, __m128d b) noexcept {
            return Blend(_mm_min_pd(b, a), b, _mm_cmpunord_pd(a, a));
        }

        static __m128d Max(__m128d a, __m128d b) noexcept {
            return Blend(_mm_max_pd(b, a), b, _mm_cmpunord_pd(a, a));
        }

        static double ReduceMin(__m128d a) noexcept {
            return _mm_cvtsd_f64(Min(a, _mm_shuffle_pd(a, a, 1)));
        }

        static double ReduceMax(__m128d a) noexcept {
            return _mm_cvtsd_f64(Max(a, _mm_shuffle_pd(a, a, 1)));
        }
    };

    template <typename T>
    struct Vec128TraitsIntBase {
        using Type = __m128i;

        static __m128i Blend(__m128i a, __m128i b, __m128i mask) noexcept {
            return _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a));
        }

        static __m128i Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const __m128i*>(data));
        }

        static T ReduceMin(__m128i a) noexcept {
            a = VecTraits<128, T>::Min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 2, 3, 2)));
            if constexpr (sizeof(T) == 8) {
                return _mm_cvtsi128_si64(a);
            } else {
                a = VecTraits<128, T>::Min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 1, 1)));
                if constexpr (sizeof(T) <= 2) {
                    a = VecTraits<128, T>::Min(a, _mm_srli_epi32(a, 16));
                    if constexpr (sizeof(T) == 1) {
                        a = VecTraits<128, T>::Min(a, _mm_srli_epi16(a, 8));
                    }
                }
                return _mm_cvtsi128_si32(a);
            }
        }

        static T ReduceMax(__m128i a) noexcept {
            a = VecTraits<128, T>::Max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 2, 3, 2)));
            if constexpr (sizeof(T) == 8) {
                return _mm_cvtsi128_si64(a);
            } else {
                a = VecTraits<128, T>::Max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 1, 1)));
                if constexpr (sizeof(T) <= 2) {
                    a = VecTraits<128, T>::Max(a, _mm_srli_epi32(a, 16));
                    if constexpr (sizeof(T) == 1) {
                        a = VecTraits<128, T>::Max(a, _mm_srli_epi16(a, 8));
                    }
                }
                return _mm_cvtsi128_si32(a);
            }
        }
    };

    template <>
    struct VecTraits<128, int8_t> : Vec128TraitsIntBase<int8_t> {
        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_min_epi8(a, b);
#else
            return Blend(a, b, _mm_cmpgt_epi8(a, b));
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_max_epi8(a, b);
#else
            return Blend(b, a, _mm_cmpgt_epi8(a, b));
#endif
        }

#if defined(__AVX512BW__) && defined(__AVX512VL__) || defined(__SSE4_1__) || defined(__AVX__)
        static int8_t ReduceMin(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_min_epi8(a);
#else
            constexpr int8_t sign_mask = 0x80;
            const __m128i to_unsigned = _mm_xor_si128(a, _mm_set1_epi8(sign_mask));
            const __m128i min128_u16 = _mm_min_epu8(to_unsigned, _mm_srli_epi16(to_unsigned, 8));
            const __m128i min16 = _mm_minpos_epu16(min128_u16);
            return _mm_cvtsi128_si32(min16) ^ sign_mask;
#endif
        }

        static int8_t ReduceMax(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_max_epi8(a);
#else
            constexpr uint8_t sign_mask = 0x7f;
            const __m128i to_opposite = _mm_xor_si128(a, _mm_set1_epi8(sign_mask));
            const __m128i max128_u16 = _mm_min_epu8(to_opposite, _mm_srli_epi16(to_opposite, 8));
            const __m128i max16 = _mm_minpos_epu16(max128_u16);
            return _mm_cvtsi128_si32(max16) ^ sign_mask;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint8_t> : Vec128TraitsIntBase<uint8_t> {
        static __m128i Min(__m128i a, __m128i b) noexcept {
            return _mm_min_epu8(a, b);
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
            return _mm_max_epu8(a, b);
        }

#if defined(__AVX512BW__) && defined(__AVX512VL__) || defined(__SSE4_1__) || defined(__AVX__)
        static uint8_t ReduceMin(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_min_epu8(a);
#else
            const __m128i min128_u16 = _mm_min_epu8(a, _mm_srli_epi16(a, 8));
            const __m128i min16 = _mm_minpos_epu16(min128_u16);
            return _mm_cvtsi128_si32(min16);
#endif
        }

        static uint8_t ReduceMax(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_max_epu8(a);
#else
            const __m128i to_1s_complement = _mm_xor_si128(a, _mm_set1_epi32(-1));
            const __m128i max128_u16 = _mm_min_epu8(to_1s_complement, _mm_srli_epi16(to_1s_complement, 8));
            const __m128i max16 = _mm_minpos_epu16(max128_u16);
            return ~_mm_cvtsi128_si32(max16);
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, int16_t> : Vec128TraitsIntBase<int16_t> {
        static __m128i Min(__m128i a, __m128i b) noexcept {
            return _mm_min_epi16(a, b);
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
            return _mm_max_epi16(a, b);
        }

#if defined(__AVX512BW__) && defined(__AVX512VL__) || defined(__SSE4_1__) || defined(__AVX__)
        static int16_t ReduceMin(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_min_epi16(a);
#else
            constexpr int16_t sign_mask = 0x8000;
            const __m128i to_unsigned = _mm_xor_si128(a, _mm_set1_epi16(sign_mask));
            const __m128i min16 = _mm_minpos_epu16(to_unsigned);
            return _mm_cvtsi128_si32(min16) ^ sign_mask;
#endif
        }

        static int16_t ReduceMax(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_max_epi16(a);
#else
            constexpr uint16_t sign_mask = 0x7fff;
            const __m128i to_opposite = _mm_xor_si128(a, _mm_set1_epi16(sign_mask));
            const __m128i max16 = _mm_minpos_epu16(to_opposite);
            return _mm_cvtsi128_si32(max16) ^ sign_mask;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint16_t> : Vec128TraitsIntBase<uint16_t> {
        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_min_epu16(a, b);
#else
            return _mm_sub_epi16(a, _mm_subs_epu16(a, b));
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_max_epu16(a, b);
#else
            return _mm_add_epi16(a, _mm_subs_epu16(b, a));
#endif
        }

#if defined(__AVX512BW__) && defined(__AVX512VL__) || defined(__SSE4_1__) || defined(__AVX__)
        static uint16_t ReduceMin(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_min_epu16(a);
#else
            return _mm_cvtsi128_si32(_mm_minpos_epu16(a));
#endif
        }

        static uint16_t ReduceMax(__m128i a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_reduce_max_epu16(a);
#else
            const __m128i to_1s_complement = _mm_xor_si128(a, _mm_set1_epi32(-1));
            const __m128i max16 = _mm_minpos_epu16(to_1s_complement);
            return ~_mm_cvtsi128_si32(max16);
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, int32_t> : Vec128TraitsIntBase<int32_t> {
        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_min_epi32(a, b);
#else
            return Blend(a, b, _mm_cmpgt_epi32(a, b));
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_max_epi32(a, b);
#else
            return Blend(b, a, _mm_cmpgt_epi32(a, b));
#endif
        }
    };

    template <>
    struct VecTraits<128, uint32_t> : Vec128TraitsIntBase<uint32_t> {
        template <bool IsMin>
        static __m128i MinMax(__m128i a, __m128i b) noexcept {
            const __m128i offset = _mm_set1_epi32(0x8000'0000);
            const __m128i a_offset = _mm_xor_si128(a, offset);
            const __m128i b_offset = _mm_xor_si128(b, offset);
            const __m128i mask = _mm_cmpgt_epi32(a_offset, b_offset);
            return IsMin ? Blend(a, b, mask) : Blend(b, a, mask);
        }

        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_min_epu32(a, b);
#else
            return MinMax<true>(a, b);
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_max_epu32(a, b);
#else
            return MinMax<false>(a, b);
#endif
        }
    };


    template <typename T>
    struct Vec128TraitsInt64Base : Vec128TraitsIntBase<T> {
#if defined(__SSE4_1__) || defined(__AVX__)
        static __m128i Blend(__m128i a, __m128i b, __m128i mask) noexcept {
            return _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b), _mm_castsi128_pd(mask)));
        }
#endif
    };

    template <>
    struct VecTraits<128, int64_t> : Vec128TraitsInt64Base<int64_t> {
        template <bool IsMin>
        static __m128i MinMax(__m128i a, __m128i b) noexcept {
#if defined(__SSE4_2__) || defined(__AVX__)
            const __m128i mask = _mm_cmpgt_epi64(a, b);
#else
            const __m128i offset = _mm_set1_epi64x(0x8000'0000);
            const __m128i a_offset = _mm_xor_si128(a, offset);
            const __m128i b_offset = _mm_xor_si128(b, offset);
            // a[0] > b[0] => gt[1] | eq[1] & gt[0]
            // a[1] > b[1] => gt[3] | eq[3] & gt[2]
            __m128i gt = _mm_cmpgt_epi32(a_offset, b_offset);
            __m128i eq = _mm_cmpeq_epi32(a_offset, b_offset);
            const __m128i gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
#if !(defined(__SSE4_1__) || defined(__AVX__))
            gt = _mm_shuffle_epi32(gt, _MM_SHUFFLE(3, 3, 1, 1));
            eq = _mm_shuffle_epi32(eq, _MM_SHUFFLE(3, 3, 1, 1));
#endif
            const __m128i mask = _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
            return IsMin ? Blend(a, b, mask) : Blend(b, a, mask);
        }

        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__AVX512VL__)
            return _mm_min_epi64(a, b);
#else
            return MinMax<true>(a, b);
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__AVX512VL__)
            return _mm_max_epi64(a, b);
#else
            return MinMax<false>(a, b);
#endif
        }
    };

    template <>
    struct VecTraits<128, uint64_t> : Vec128TraitsInt64Base<uint64_t> {
        template <bool IsMin>
        static __m128i MinMax(__m128i a, __m128i b) noexcept {
            const __m128i offset = _mm_set1_epi64x(0x8000'0000'0000'0000);
            const __m128i a_offset = _mm_xor_si128(a, offset);
            const __m128i b_offset = _mm_xor_si128(b, offset);
#if defined(__SSE4_2__) || defined(__AVX__)
            const __m128i mask = _mm_cmpgt_epi64(a_offset, b_offset);
#else
            // a[0] > b[0] => gt[1] | eq[1] & gt[0]
            // a[1] > b[1] => gt[3] | eq[3] & gt[2]
            __m128i gt = _mm_cmpgt_epi32(a_offset, b_offset);
            __m128i eq = _mm_cmpeq_epi32(a_offset, b_offset);
            const __m128i gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
#if !(defined(__SSE4_1__) || defined(__AVX__))
            gt = _mm_shuffle_epi32(gt, _MM_SHUFFLE(3, 3, 1, 1));
            eq = _mm_shuffle_epi32(eq, _MM_SHUFFLE(3, 3, 1, 1));
#endif
            const __m128i mask = _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
            return IsMin ? Blend(a, b, mask) : Blend(b, a, mask);
        }

        static __m128i Min(__m128i a, __m128i b) noexcept {
#if defined(__AVX512VL__)
            return _mm_min_epu64(a, b);
#else
            return MinMax<true>(a, b);
#endif
        }

        static __m128i Max(__m128i a, __m128i b) noexcept {
#if defined(__AVX512VL__)
            return _mm_max_epu64(a, b);
#else
            return MinMax<false>(a, b);
#endif
        }
    };
#endif
// ===================================== Vec128 =========================================

// ===================================== Vec256 =========================================
#ifdef __AVX__
    template <>
    struct VecTraits<256, float> {
        using Type = __m256;

        static __m256 Load(const void* data) noexcept {
            return _mm256_loadu_ps(static_cast<const float*>(data));
        }

        static __m256 Min(__m256 a, __m256 b) noexcept {
            return _mm256_blendv_ps(_mm256_min_ps(b, a), b, _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
        }

        static __m256 Max(__m256 a, __m256 b) noexcept {
            return _mm256_blendv_ps(_mm256_max_ps(b, a), b, _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
        }

        static float ReduceMin(__m256 a) noexcept {
            const __m128 low = _mm256_castps256_ps128(a);
            const __m128 high = _mm256_extractf128_ps(a, 1);
            const __m128 v = VecTraits<128, float>::Min(low, high);
            return VecTraits<128, float>::ReduceMin(v);
        }

        static float ReduceMax(__m256 a) noexcept {
            const __m128 low = _mm256_castps256_ps128(a);
            const __m128 high = _mm256_extractf128_ps(a, 1);
            const __m128 v = VecTraits<128, float>::Max(low, high);
            return VecTraits<128, float>::ReduceMax(v);
        }
    };

    template <>
    struct VecTraits<256, double> {
        using Type = __m256d;

        static __m256d Load(const void* data) noexcept {
            return _mm256_loadu_pd(static_cast<const double*>(data));
        }

        static __m256d Min(__m256d a, __m256d b) noexcept {
            return _mm256_blendv_pd(_mm256_min_pd(b, a), b, _mm256_cmp_pd(a, a, _CMP_UNORD_Q));
        }

        static __m256d Max(__m256d a, __m256d b) noexcept {
            return _mm256_blendv_pd(_mm256_max_pd(b, a), b, _mm256_cmp_pd(a, a, _CMP_UNORD_Q));
        }

        static double ReduceMin(__m256d a) noexcept {
            const __m128d low = _mm256_castpd256_pd128(a);
            const __m128d high = _mm256_extractf128_pd(a, 1);
            const __m128d v = VecTraits<128, double>::Min(low, high);
            return VecTraits<128, double>::ReduceMin(v);
        }

        static double ReduceMax(__m256d a) noexcept {
            const __m128d low = _mm256_castpd256_pd128(a);
            const __m128d high = _mm256_extractf128_pd(a, 1);
            const __m128d v = VecTraits<128, double>::Max(low, high);
            return VecTraits<128, double>::ReduceMax(v);
        }
    };

    template <typename T>
    struct Vec256TraitsIntBase {
        using Type = __m256i;

        static __m256i Load(const void* data) noexcept {
            return _mm256_loadu_si256(static_cast<const __m256i*>(data));
        }

        static __m256i Min(__m256i a, __m256i b) noexcept {
            const __m128i a_low = _mm256_castsi256_si128(a);
            const __m128i b_low = _mm256_castsi256_si128(b);
            const __m128i a_high = _mm256_extractf128_si256(a, 1);
            const __m128i b_high = _mm256_extractf128_si256(b, 1);
            const __m128i low = VecTraits<128, T>::Min(a_low, b_low);
            const __m128i high = VecTraits<128, T>::Min(a_high, b_high);
            return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            const __m128i a_low = _mm256_castsi256_si128(a);
            const __m128i b_low = _mm256_castsi256_si128(b);
            const __m128i a_high = _mm256_extractf128_si256(a, 1);
            const __m128i b_high = _mm256_extractf128_si256(b, 1);
            const __m128i low = VecTraits<128, T>::Max(a_low, b_low);
            const __m128i high = VecTraits<128, T>::Max(a_high, b_high);
            return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
        }

        static T ReduceMin(__m256i a) noexcept {
            const __m128i low = _mm256_castsi256_si128(a);
            const __m128i high = _mm256_extractf128_si256(a, 1);
            const __m128i v = VecTraits<128, T>::Min(low, high);
            return VecTraits<128, T>::ReduceMin(v);
        }

        static T ReduceMax(__m256i a) noexcept {
            const __m128i low = _mm256_castsi256_si128(a);
            const __m128i high = _mm256_extractf128_si256(a, 1);
            const __m128i v = VecTraits<128, T>::Max(low, high);
            return VecTraits<128, T>::ReduceMax(v);
        }
    };

    template <>
    struct VecTraits<256, int8_t> : Vec256TraitsIntBase<int8_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epi8(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epi8(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        static int8_t ReduceMin(__m256i a) noexcept {
            return _mm256_reduce_min_epi8(a);
        }

        static int8_t ReduceMax(__m256i a) noexcept {
            return _mm256_reduce_max_epi8(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint8_t> : Vec256TraitsIntBase<uint8_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epu8(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epu8(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        static uint8_t ReduceMin(__m256i a) noexcept {
            return _mm256_reduce_min_epu8(a);
        }

        static uint8_t ReduceMax(__m256i a) noexcept {
            return _mm256_reduce_max_epu8(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, int16_t> : Vec256TraitsIntBase<int16_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epi16(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epi16(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        static int16_t ReduceMin(__m256i a) noexcept {
            return _mm256_reduce_min_epi16(a);
        }

        static int16_t ReduceMax(__m256i a) noexcept {
            return _mm256_reduce_max_epi16(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint16_t> : Vec256TraitsIntBase<uint16_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epu16(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epu16(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        static uint16_t ReduceMin(__m256i a) noexcept {
            return _mm256_reduce_min_epu16(a);
        }

        static uint16_t ReduceMax(__m256i a) noexcept {
            return _mm256_reduce_max_epu16(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, int32_t> : Vec256TraitsIntBase<int32_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epi32(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epi32(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint32_t> : Vec256TraitsIntBase<uint32_t> {
#ifdef __AVX2__
        static __m256i Min(__m256i a, __m256i b) noexcept {
            return _mm256_min_epu32(a, b);
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
            return _mm256_max_epu32(a, b);
        }
#endif
    };

    template <typename T>
    struct Vec256TraitsInt64Base : Vec256TraitsIntBase<T> {
        static __m256i Blend(__m256i a, __m256i b, __m256i mask) noexcept {
            return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), _mm256_castsi256_pd(mask)));
        }
    };

    template <>
    struct VecTraits<256, int64_t> : Vec256TraitsInt64Base<int64_t> {
        static __m256i Min(__m256i a, __m256i b) noexcept {
#ifdef __AVX512VL__
            return _mm256_min_epi64(a, b);
#else
            return Blend(a, b, _mm256_cmpgt_epi64(a, b));
#endif
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
#ifdef __AVX512VL__
            return _mm256_max_epi64(a, b);
#else
            return Blend(b, a, _mm256_cmpgt_epi64(a, b));
#endif
        }
    };

    template <>
    struct VecTraits<256, uint64_t> : Vec256TraitsInt64Base<uint64_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static __m256i MinMax(__m256i a, __m256i b) noexcept {
            const __m256i offset = _mm256_set1_epi64x(0x8000'0000'0000'0000);
            const __m256i a_offset = _mm256_xor_si256(a, offset);
            const __m256i b_offset = _mm256_xor_si256(b, offset);
            const __m256i mask = _mm256_cmpgt_epi64(a_offset, b_offset);
            return IsMin ? Blend(a, b, mask) : Blend(b, a, mask);
        }

        static __m256i Min(__m256i a, __m256i b) noexcept {
#ifdef __AVX512VL__
            return _mm256_min_epu64(a, b);
#else
            return MinMax<true>(a, b);
#endif
        }

        static __m256i Max(__m256i a, __m256i b) noexcept {
#ifdef __AVX512VL__
            return _mm256_max_epu64(a, b);
#else
            return MinMax<false>(a, b);
#endif
        }
#endif
    };
#endif
// ===================================== Vec256 =========================================

// ===================================== Vec512 =========================================
#ifdef __AVX512F__
    template <>
    struct VecTraits<512, float> {
        using Type = __m512;

        static __m512 Load(const void* data) noexcept {
            return _mm512_loadu_ps(data);
        }

        static __m512 Min(__m512 a, __m512 b) noexcept {
            return _mm512_min_ps(a, b);
        }

        static __m512 Max(__m512 a, __m512 b) noexcept {
            return _mm512_max_ps(a, b);
        }

        static float ReduceMin(__m512 a) noexcept {
            return _mm512_reduce_min_ps(a);
        }

        static float ReduceMax(__m512 a) noexcept {
            return _mm512_reduce_max_ps(a);
        }
    };

    template <>
    struct VecTraits<512, double> {
        static __m512d Load(const void* data) noexcept {
            return _mm512_loadu_pd(data);
        }

        static __m512d Min(__m512d a, __m512d b) noexcept {
            return _mm512_min_pd(a, b);
        }

        static __m512d Max(__m512d a, __m512d b) noexcept {
            return _mm512_max_pd(a, b);
        }

        static double ReduceMin(__m512d a) noexcept {
            return _mm512_reduce_min_pd(a);
        }

        static double ReduceMax(__m512d a) noexcept {
            return _mm512_reduce_max_pd(a);
        }
    };

    template <typename T>
    struct Vec512TraitsIntBase {
        using Type = __m512i;

        static __m512i Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }

        static __m512i Min(__m512i a, __m512i b) noexcept {
            const __m256i a_low = _mm512_castsi512_si256(a);
            const __m256i b_low = _mm512_castsi512_si256(b);
            const __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
            const __m256i b_high = _mm512_extracti64x4_epi64(b, 1);
            const __m256i low = VecTraits<256, T>::Min(a_low, b_low);
            const __m256i high = VecTraits<256, T>::Min(a_high, b_high);
            return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            const __m256i a_low = _mm512_castsi512_si256(a);
            const __m256i b_low = _mm512_castsi512_si256(b);
            const __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
            const __m256i b_high = _mm512_extracti64x4_epi64(b, 1);
            const __m256i low = VecTraits<256, T>::Max(a_low, b_low);
            const __m256i high = VecTraits<256, T>::Max(a_high, b_high);
            return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
        }

        static T ReduceMin(__m512i a) noexcept {
            const __m256i low = _mm512_castsi512_si256(a);
            const __m256i high = _mm512_extracti64x4_epi64(a, 1);
            const __m256i v = VecTraits<256, T>::Min(low, high);
            return VecTraits<256, T>::ReduceMin(v);
        }

        static T ReduceMax(__m512i a) noexcept {
            const __m256i low = _mm512_castsi512_si256(a);
            const __m256i high = _mm512_extracti64x4_epi64(a, 1);
            const __m256i v = VecTraits<256, T>::Max(low, high);
            return VecTraits<256, T>::ReduceMax(v);
        }
    };

    template <>
    struct VecTraits<512, int8_t> : Vec512TraitsIntBase<int8_t> {
#ifdef __AVX512BW__
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epi8(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epi8(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, uint8_t> : Vec512TraitsIntBase<uint8_t> {
#ifdef __AVX512BW__
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epu8(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epu8(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int16_t> : Vec512TraitsIntBase<int16_t> {
#ifdef __AVX512BW__
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epi16(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epi16(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, uint16_t> : Vec512TraitsIntBase<uint16_t> {
#ifdef __AVX512BW__
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epu16(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epu16(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int32_t> : Vec512TraitsIntBase<int32_t> {
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epi32(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epi32(a, b);
        }

        static int32_t ReduceMin(__m512i a) noexcept {
            return _mm512_reduce_min_epi32(a);
        }

        static int32_t ReduceMax(__m512i a) noexcept {
            return _mm512_reduce_max_epi32(a);
        }
    };

    template <>
    struct VecTraits<512, uint32_t> : Vec512TraitsIntBase<uint32_t> {
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epu32(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epu32(a, b);
        }

        static uint32_t ReduceMin(__m512i a) noexcept {
            return _mm512_reduce_min_epu32(a);
        }

        static uint32_t ReduceMax(__m512i a) noexcept {
            return _mm512_reduce_max_epu32(a);
        }
    };

    template <>
    struct VecTraits<512, int64_t> : Vec512TraitsIntBase<int64_t> {
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epi64(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epi64(a, b);
        }

        static int64_t ReduceMin(__m512i a) noexcept {
            return _mm512_reduce_min_epi64(a);
        }

        static int64_t ReduceMax(__m512i a) noexcept {
            return _mm512_reduce_max_epi64(a);
        }
    };

    template <>
    struct VecTraits<512, uint64_t> : Vec512TraitsIntBase<uint64_t> {
        static __m512i Min(__m512i a, __m512i b) noexcept {
            return _mm512_min_epu64(a, b);
        }

        static __m512i Max(__m512i a, __m512i b) noexcept {
            return _mm512_max_epu64(a, b);
        }

        static uint64_t ReduceMin(__m512i a) noexcept {
            return _mm512_reduce_min_epu64(a);
        }

        static uint64_t ReduceMax(__m512i a) noexcept {
            return _mm512_reduce_max_epu64(a);
        }
    };
#endif
    // ===================================== Vec512 =========================================

    enum MinMaxMode {
        ModeMin = 1,
        ModeMax = 2,
        ModeMinMax = ModeMin | ModeMax
    };

    template <MinMaxMode Mode, typename T, class Traits>
    auto OneVecImpl(const void* data) noexcept {
        using VecTy = typename Traits::Type;

        auto v = Traits::Load(data);

        T min_val, max_val;
        if constexpr (Mode & ModeMin) {
            min_val = static_cast<T>(Traits::ReduceMin(v));
        }
        if constexpr (Mode & ModeMax) {
            max_val = static_cast<T>(Traits::ReduceMax(v));
        }

        if constexpr (Mode == ModeMin) {
            return min_val;
        } else if constexpr (Mode == ModeMax) {
            return max_val;
        } else {
            return std::pair{min_val, max_val};
        }
    }

    template <MinMaxMode Mode, typename T, class Traits>
    auto VecLoopImpl(const void* begin, const void* end) noexcept {
        using VecTy = typename Traits::Type;

        const auto* first = static_cast<const VecTy*>(begin);
        const auto* const last = static_cast<const VecTy*>(end);

        auto v_min = Traits::Load(first);
        auto v_max = v_min;
        ++first;

        for (; first + 1 <= last; ++first) {
            const auto v = Traits::Load(first);
            if constexpr (Mode & ModeMin) {
                v_min = Traits::Min(v_min, v);
            }
            if constexpr (Mode & ModeMax) {
                v_max = Traits::Max(v_max, v);
            }
        }
        if (first != last) {
            const auto v = Traits::Load(last - 1);
            if constexpr (Mode & ModeMin) {
                v_min = Traits::Min(v_min, v);
            }
            if constexpr (Mode & ModeMax) {
                v_max = Traits::Max(v_max, v);
            }
        }

        T min_val, max_val;
        if constexpr (Mode & ModeMin) {
            min_val = static_cast<T>(Traits::ReduceMin(v_min));
        }
        if constexpr (Mode & ModeMax) {
            max_val = static_cast<T>(Traits::ReduceMax(v_max));
        }

        if constexpr (Mode == ModeMin) {
            return min_val;
        } else if constexpr (Mode == ModeMax) {
            return max_val;
        } else {
            return std::pair{min_val, max_val};
        }
    }

    template <MinMaxMode Mode, typename T>
    auto ScalarImpl(const T* begin, const T* end) noexcept {
        T min_val = *begin;
        T max_val = min_val;
        ++begin;
        for (; begin != end; ++begin) {
            const T val = *begin;
            if constexpr (Mode & ModeMin) {
                if (val < min_val) {
                    min_val = val;
                }
            }
            if constexpr (Mode & ModeMax) {
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        if constexpr (Mode == ModeMin) {
            return min_val;
        } else if constexpr (Mode == ModeMax) {
            return max_val;
        } else {
            return std::pair{min_val, max_val};
        }
    }

    template <MinMaxMode Mode, VectorizeSafely T>
    auto Helper(const T* data, size_t length) noexcept {
        using ElemTy = decltype(ToArithmeticType<T>());
        const T* end = data + length;
#ifdef __AVX512F__
#ifndef __AVX512BW__
        if constexpr (std::integral<ElemTy> && sizeof(ElemTy) > 2)
#endif
        {
            using Vec512Traits = VecTraits<512, ElemTy>;
            if (length > 64 / sizeof(ElemTy)) {
                return VecLoopImpl<Mode, T, Vec512Traits>(data, end);
            }
            if (length == 64 / sizeof(ElemTy)) {
                return OneVecImpl<Mode, T, Vec512Traits>(data);
            }
        }
#endif
#ifdef __AVX__
        using Vec256Traits = VecTraits<256, ElemTy>;
        if (length > 32 / sizeof(ElemTy)) {
            return VecLoopImpl<Mode, T, Vec256Traits>(data, end);
        }
        if (length == 32 / sizeof(ElemTy)) {
            return OneVecImpl<Mode, T, Vec256Traits>(data);
        }
#endif
#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
        using Vec128Traits = VecTraits<128, ElemTy>;
        if (length > 16 / sizeof(ElemTy)) {
            return VecLoopImpl<Mode, T, Vec128Traits>(data, end);
        }
        if (length == 16 / sizeof(ElemTy)) {
            return OneVecImpl<Mode, T, Vec128Traits>(data);
        }
#endif
        return ScalarImpl<Mode>(data, end);
    }
}// namespace detail


template <detail::VectorizeSafely T>
T Min(const T* data, size_t length) noexcept {
    return detail::Helper<detail::ModeMin>(data, length);
}

template <detail::VectorizeSafely T>
T Max(const T* data, size_t length) noexcept {
    return detail::Helper<detail::ModeMax>(data, length);
}

template <detail::VectorizeSafely T>
std::pair<T, T> MinMax(const T* data, size_t length) noexcept {
    return detail::Helper<detail::ModeMinMax>(data, length);
}
