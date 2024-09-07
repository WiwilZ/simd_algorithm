#pragma once


#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cstddef>
#include <cstdint>

#include <bit>
#include <concepts>
#include <utility>


namespace detail {
    enum MinMaxMode {
        ModeMin = 1,
        ModeMax = 2,
        ModeMinMax = ModeMin | ModeMax
    };

    template <size_t VecBits, typename ElemType>
    struct VecTraits;

    // _mm128/256/512_min/max_ps/d(a, b) returns either a or b is NaN ? b : min/max(a, b)
    // _mm128/256/512_cmpunord_ps/d(a, a) returns true if a is NaN otherwise false

#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
    template <>
    struct VecTraits<128, float> {
        using Type = __m128;

        static Type Conditional(Type mask, Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_blendv_ps(b, a, mask);
#else
            return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
#endif
        }

        static Type Load(const void* data) noexcept {
            return _mm_loadu_ps(static_cast<const float*>(data));
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return Conditional(_mm_cmpunord_ps(a, a), b, IsMin ? _mm_min_ps(b, a) : _mm_max_ps(b, a));
        }

        template <bool IsMin>
        static float ReduceMinMax(Type a) noexcept {
            const auto f = MinMax<IsMin>;
            a = f(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 3, 2)));
            a = f(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
            return _mm_cvtss_f32(a);
        }
    };

    template <>
    struct VecTraits<128, double> {
        using Type = __m128d;

        static Type Conditional(Type mask, Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return _mm_blendv_pd(b, a, mask);
#else
            return _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b));
#endif
        }

        static Type Load(const void* data) noexcept {
            return _mm_loadu_pd(static_cast<const double*>(data));
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return Conditional(_mm_cmpunord_pd(a, a), b, IsMin ? _mm_min_pd(b, a) : _mm_max_pd(b, a));
        }

        template <bool IsMin>
        static double ReduceMinMax(Type a) noexcept {
            return _mm_cvtsd_f64(MinMax<IsMin>(a, _mm_shuffle_pd(a, a, 1)));
        }
    };


    template <typename T>
    struct Vec128iTraitsBase {
        using Type = __m128i;

        static Type Conditional(Type mask, Type a, Type b) noexcept {
            return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));
        }

        static Type Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const Type*>(data));
        }

        template <bool IsMin>
        static T ReduceMinMax(Type a) noexcept {
            a = VecTraits<128, T>::template MinMax<IsMin>(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 2, 3, 2)));
            if constexpr (sizeof(T) == 8) {
                return _mm_cvtsi128_si64(a);
            } else {
                a = VecTraits<128, T>::template MinMax<IsMin>(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 1, 1)));
                if constexpr (sizeof(T) <= 2) {
                    a = VecTraits<128, T>::template MinMax<IsMin>(a, _mm_srli_epi32(a, 16));
                    if constexpr (sizeof(T) == 1) {
                        a = VecTraits<128, T>::template MinMax<IsMin>(a, _mm_srli_epi16(a, 8));
                    }
                }
                return _mm_cvtsi128_si32(a);
            }
        }
    };

    template <>
    struct VecTraits<128, int8_t> : Vec128iTraitsBase<int8_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return IsMin ? _mm_min_epi8(a, b) : _mm_max_epi8(a, b);
#else
            const auto mask = _mm_cmpgt_epi8(a, b);
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }

#if defined(__SSE4_1__) || defined(__AVX__)
        template <bool IsMin>
        static int8_t ReduceMinMax(Type a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return IsMin ? _mm_reduce_min_epi8(a) : _mm_reduce_max_epi8(a);
#else
            constexpr int8_t sign_mask = IsMin ? 0x80 : 0x7f;
            static const auto mask = _mm_set1_epi8(sign_mask);
            a = _mm_xor_si128(a, mask);
            a = _mm_min_epu8(a, _mm_srli_epi16(a, 8));
            return _mm_cvtsi128_si32(_mm_minpos_epu16(a)) ^ sign_mask;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint8_t> : Vec128iTraitsBase<uint8_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm_min_epu8(a, b) : _mm_max_epu8(a, b);
        }

#if defined(__SSE4_1__) || defined(__AVX__)
        template <bool IsMin>
        static uint8_t ReduceMinMax(Type a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return IsMin ? _mm_reduce_min_epu8(a) : _mm_reduce_max_epu8(a);
#else
            if constexpr (!IsMin) {
                static const auto mask = _mm_set1_epi32(-1);
                a = _mm_xor_si128(a, mask);
            }
            a = _mm_min_epu8(a, _mm_srli_epi16(a, 8));
            const uint8_t res = _mm_cvtsi128_si32(_mm_minpos_epu16(a));
            return IsMin ? res : ~res;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, int16_t> : Vec128iTraitsBase<int16_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm_min_epi16(a, b) : _mm_max_epi16(a, b);
        }

#if defined(__SSE4_1__) || defined(__AVX__)
        template <bool IsMin>
        static int16_t ReduceMinMax(Type a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return IsMin ? _mm_reduce_min_epi16(a) : _mm_reduce_max_epi16(a);
#else
            constexpr int16_t sign_mask = IsMin ? 0x8000 : 0x7fff;
            static const auto mask = _mm_set1_epi16(sign_mask);
            a = _mm_xor_si128(a, mask);
            return _mm_cvtsi128_si32(_mm_minpos_epu16(a)) ^ sign_mask;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint16_t> : Vec128iTraitsBase<uint16_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return IsMin ? _mm_min_epu16(a, b) : _mm_max_epu16(a, b);
#else
            return IsMin ? _mm_sub_epi16(a, _mm_subs_epu16(a, b)) : _mm_add_epi16(a, _mm_subs_epu16(b, a));
#endif
        }

#if defined(__SSE4_1__) || defined(__AVX__)
        template <bool IsMin>
        static uint16_t ReduceMinMax(Type a) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return IsMin ? _mm_reduce_min_epu16(a) : _mm_reduce_max_epu16(a);
#else
            if constexpr (!IsMin) {
                static const auto mask = _mm_set1_epi32(-1);
                a = _mm_xor_si128(a, mask);
            }
            const uint16_t res = _mm_cvtsi128_si32(_mm_minpos_epu16(a));
            return IsMin ? res : ~res;
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, int32_t> : Vec128iTraitsBase<int32_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return IsMin ? _mm_min_epi32(a, b) : _mm_max_epi32(a, b);
#else
            const auto mask = _mm_cmpgt_epi32(a, b);
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
    };

    template <>
    struct VecTraits<128, uint32_t> : Vec128iTraitsBase<uint32_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return IsMin ? _mm_min_epu32(a, b) : _mm_max_epu32(a, b);
#else
            static const auto offset = _mm_set1_epi32(0x8000'0000);
            const auto x = _mm_xor_si128(a, offset);
            const auto y = _mm_xor_si128(b, offset);
            const auto mask = _mm_cmpgt_epi32(x, y);
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
    };


    template <typename T>
    struct Vec128Int64TraitsBase : Vec128iTraitsBase<T> {
        using typename Vec128iTraitsBase<T>::Type;

#if defined(__SSE4_1__) || defined(__AVX__)
        static Type Conditional(Type mask, Type a, Type b) noexcept {
            return _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(b), _mm_castsi128_pd(a), _mm_castsi128_pd(mask)));
        }
#endif
    };

    template <>
    struct VecTraits<128, int64_t> : Vec128Int64TraitsBase<int64_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return IsMin ? _mm_min_epi64(a, b) : _mm_max_epi64(a, b);
#elif defined(__AVX512F__)
            const auto x = _mm512_castsi128_si512(a);
            const auto y = _mm512_castsi128_si512(b);
            return _mm512_castsi512_si128(IsMin ? _mm512_min_epi64(x, y) : _mm512_min_epi64(x, y));
#else
#if defined(__SSE4_2__) || defined(__AVX__)
            const auto mask = _mm_cmpgt_epi64(a, b);
#else
            static const auto offset = _mm_set1_epi64x(0x8000'0000);
            const auto x = _mm_xor_si128(a, offset);
            const auto y = _mm_xor_si128(b, offset);
            // a[0] > b[0] => gt[1] | eq[1] & gt[0]
            // a[1] > b[1] => gt[3] | eq[3] & gt[2]
            auto gt = _mm_cmpgt_epi32(x, y);
            auto eq = _mm_cmpeq_epi32(x, y);
            const auto gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
#ifndef __SSE4_1__
            gt = _mm_shuffle_epi32(gt, _MM_SHUFFLE(3, 3, 1, 1));
            eq = _mm_shuffle_epi32(eq, _MM_SHUFFLE(3, 3, 1, 1));
#endif
            const auto mask = _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
    };

    template <>
    struct VecTraits<128, uint64_t> : Vec128Int64TraitsBase<uint64_t> {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return IsMin ? _mm_min_epu64(a, b) : _mm_max_epu64(a, b);
#elif defined(__AVX512F__)
            const auto x = _mm512_castsi128_si512(a);
            const auto y = _mm512_castsi128_si512(b);
            return _mm512_castsi512_si128(IsMin ? _mm512_min_epu64(x, y) : _mm512_min_epu64(x, y));
#else
            static const auto offset = _mm_set1_epi64x(0x8000'0000'0000'0000);
            const auto x = _mm_xor_si128(a, offset);
            const auto y = _mm_xor_si128(b, offset);
#if defined(__SSE4_2__) || defined(__AVX__)
            const auto mask = _mm_cmpgt_epi64(x, y);
#else
            // a[0] > b[0] => gt[1] | eq[1] & gt[0]
            // a[1] > b[1] => gt[3] | eq[3] & gt[2]
            auto gt = _mm_cmpgt_epi32(x, y);
            auto eq = _mm_cmpeq_epi32(x, y);
            const auto gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
#ifndef __SSE4_1__
            gt = _mm_shuffle_epi32(gt, _MM_SHUFFLE(3, 3, 1, 1));
            eq = _mm_shuffle_epi32(eq, _MM_SHUFFLE(3, 3, 1, 1));
#endif
            const auto mask = _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
    };
#endif


#ifdef __AVX__
    template <>
    struct VecTraits<256, float> {
        using Type = __m256;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_ps(static_cast<const float*>(data));
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return _mm256_blendv_ps(IsMin ? _mm256_min_ps(b, a) : _mm256_max_ps(b, a), b, _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
        }

        template <bool IsMin>
        static float ReduceMinMax(Type a) noexcept {
            const auto low = _mm256_castps256_ps128(a);
            const auto high = _mm256_extractf128_ps(a, 1);
            return VecTraits<128, float>::template ReduceMinMax<IsMin>(VecTraits<128, float>::template MinMax<IsMin>(low, high));
        }
    };

    template <>
    struct VecTraits<256, double> {
        using Type = __m256d;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_pd(static_cast<const double*>(data));
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return _mm256_blendv_pd(IsMin ? _mm256_min_pd(b, a) : _mm256_max_pd(b, a), b, _mm256_cmp_pd(a, a, _CMP_UNORD_Q));
        }

        template <bool IsMin>
        static double ReduceMinMax(Type a) noexcept {
            const auto low = _mm256_castpd256_pd128(a);
            const auto high = _mm256_extractf128_pd(a, 1);
            return VecTraits<128, double>::template ReduceMinMax<IsMin>(VecTraits<128, double>::template MinMax<IsMin>(low, high));
        }
    };


    template <typename T>
    struct Vec256iTraitsBase {
        using Type = __m256i;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_si256(static_cast<const Type*>(data));
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            const auto a_low = _mm256_castsi256_si128(a);
            const auto b_low = _mm256_castsi256_si128(b);
            const auto a_high = _mm256_extractf128_si256(a, 1);
            const auto b_high = _mm256_extractf128_si256(b, 1);
            const auto low = VecTraits<128, T>::template MinMax<IsMin>(a_low, b_low);
            const auto high = VecTraits<128, T>::template MinMax<IsMin>(a_high, b_high);
            return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
        }

        template <bool IsMin>
        static T ReduceMinMax(Type a) noexcept {
            const auto low = _mm256_castsi256_si128(a);
            const auto high = _mm256_extractf128_si256(a, 1);
            return VecTraits<128, T>::template ReduceMinMax<IsMin>(VecTraits<128, T>::template MinMax<IsMin>(low, high));
        }
    };

    template <>
    struct VecTraits<256, int8_t> : Vec256iTraitsBase<int8_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epi8(a, b) : _mm256_max_epi8(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        template <bool IsMin>
        static int8_t ReduceMinMax(Type a) noexcept {
            return IsMin ? _mm256_reduce_min_epi8(a) : _mm256_reduce_max_epi8(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint8_t> : Vec256iTraitsBase<uint8_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epu8(a, b) : _mm256_max_epu8(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        template <bool IsMin>
        static uint8_t ReduceMinMax(Type a) noexcept {
            return IsMin ? _mm256_reduce_min_epu8(a) : _mm256_reduce_max_epu8(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, int16_t> : Vec256iTraitsBase<int16_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epi16(a, b) : _mm256_max_epi16(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        template <bool IsMin>
        static int16_t ReduceMinMax(Type a) noexcept {
            return IsMin ? _mm256_reduce_min_epi16(a) : _mm256_reduce_max_epi16(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint16_t> : Vec256iTraitsBase<uint16_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epu16(a, b) : _mm256_max_epu16(a, b);
        }
#endif

#if defined(__AVX512BW__) && defined(__AVX512VL__)
        template <bool IsMin>
        static uint16_t ReduceMinMax(Type a) noexcept {
            return IsMin ? _mm256_reduce_min_epu16(a) : _mm256_reduce_max_epu16(a);
        }
#endif
    };

    template <>
    struct VecTraits<256, int32_t> : Vec256iTraitsBase<int32_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epi32(a, b) : _mm256_max_epi32(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<256, uint32_t> : Vec256iTraitsBase<uint32_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm256_min_epu32(a, b) : _mm256_max_epu32(a, b);
        }
#endif
    };

    template <typename T>
    struct Vec256Int64TraitsBase : Vec256iTraitsBase<T> {
        using typename Vec256iTraitsBase<T>::Type;

        static Type Conditional(Type mask, Type a, Type b) noexcept {
            return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(b), _mm256_castsi256_pd(a), _mm256_castsi256_pd(mask)));
        }
    };

    template <>
    struct VecTraits<256, int64_t> : Vec256Int64TraitsBase<int64_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return IsMin ? _mm256_min_epi64(a, b) : _mm256_max_epi64(a, b);
#else
            const auto mask = _mm256_cmpgt_epi64(a, b);
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, uint64_t> : Vec256Int64TraitsBase<uint64_t> {
#ifdef __AVX2__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return IsMin ? _mm256_min_epu64(a, b) : _mm256_max_epu64(a, b);
#elif defined(__AVX512F__)
            const auto x = _mm512_castsi256_si512(a);
            const auto y = _mm512_castsi256_si512(b);
            return _mm512_castsi512_si256(IsMin ? _mm512_min_epu64(x, y) : _mm512_min_epu64(x, y));
#else
            static const auto offset = _mm256_set1_epi64x(0x8000'0000'0000'0000);
            const auto x = _mm256_xor_si256(a, offset);
            const auto y = _mm256_xor_si256(b, offset);
            const auto mask = _mm256_cmpgt_epi64(x, y);
            return IsMin ? Conditional(mask, b, a) : Conditional(mask, a, b);
#endif
        }
#endif
    };
#endif


#ifdef __AVX512F__
    template <>
    struct VecTraits<512, float> {
        using Type = __m512;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_ps(data);
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return _mm512_mask_blend_ps(_mm512_cmpunord_ps_mask(a, a), IsMin ? _mm512_min_ps(b, a) : _mm512_max_ps(b, a), b);
        }

        template <bool IsMin>
        static float ReduceMinMax(Type a) noexcept {
            const auto low = _mm512_castps512_ps256(a);
            const auto high = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(a), 1));
            return VecTraits<256, float>::template ReduceMinMax<IsMin>(VecTraits<256, float>::template MinMax<IsMin>(low, high));
        }
    };

    template <>
    struct VecTraits<512, double> {
        using Type = __m512d;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_pd(data);
        }

        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return _mm512_mask_blend_pd(_mm512_cmpunord_pd_mask(a, a), IsMin ? _mm512_min_pd(b, a) : _mm512_max_pd(b, a), b);
        }

        template <bool IsMin>
        static double ReduceMinMax(Type a) noexcept {
            const auto low = _mm512_castpd512_pd256(a);
            const auto high = _mm512_extractf64x4_pd(a, 1);
            return VecTraits<256, double>::template ReduceMinMax<IsMin>(VecTraits<256, double>::template MinMax<IsMin>(low, high));
        }
    };


    struct Vec512iTraitsBase {
        using Type = __m512i;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }
    };

    template <typename T>
    struct Vec512iTraitsBase1 : Vec512iTraitsBase {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            const auto a_low = _mm512_castsi512_si256(a);
            const auto b_low = _mm512_castsi512_si256(b);
            const auto a_high = _mm512_extracti64x4_epi64(a, 1);
            const auto b_high = _mm512_extracti64x4_epi64(b, 1);
            const auto low = VecTraits<256, T>::template MinMax<IsMin>(a_low, b_low);
            const auto high = VecTraits<256, T>::template MinMax<IsMin>(a_high, b_high);
            return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
        }

        template <bool IsMin>
        static T ReduceMinMax(Type a) noexcept {
            const auto low = _mm512_castsi512_si256(a);
            const auto high = _mm512_extracti64x4_epi64(a, 1);
            return VecTraits<256, T>::template ReduceMinMax<IsMin>(VecTraits<256, T>::template MinMax<IsMin>(low, high));
        }
    };


    template <>
    struct VecTraits<512, int8_t> : Vec512iTraitsBase1<int8_t> {
#ifdef __AVX512BW__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epi8(a, b) : _mm512_max_epi8(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, uint8_t> : Vec512iTraitsBase1<uint8_t> {
#ifdef __AVX512BW__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epu8(a, b) : _mm512_max_epu8(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int16_t> : Vec512iTraitsBase1<int16_t> {
#ifdef __AVX512BW__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epi16(a, b) : _mm512_max_epi16(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, uint16_t> : Vec512iTraitsBase1<uint16_t> {
#ifdef __AVX512BW__
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epu16(a, b) : _mm512_max_epu16(a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int32_t> : Vec512iTraitsBase {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epi32(a, b) : _mm512_max_epi32(a, b);
        }

        template <bool IsMin>
        static int32_t ReduceMin(Type a, Type b) noexcept {
            return IsMin ? _mm512_reduce_min_epi32(a) : _mm512_reduce_max_epi32(a);
        }
    };

    template <>
    struct VecTraits<512, uint32_t> : Vec512iTraitsBase {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epu32(a, b) : _mm512_max_epu32(a, b);
        }

        template <bool IsMin>
        static uint32_t ReduceMin(Type a, Type b) noexcept {
            return IsMin ? _mm512_reduce_min_epu32(a) : _mm512_reduce_max_epu32(a);
        }
    };

    template <>
    struct VecTraits<512, int64_t> : Vec512iTraitsBase {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epi64(a, b) : _mm512_max_epi64(a, b);
        }

        template <bool IsMin>
        static int64_t ReduceMin(Type a, Type b) noexcept {
            return IsMin ? _mm512_reduce_min_epi64(a) : _mm512_reduce_max_epi64(a);
        }
    };

    template <>
    struct VecTraits<512, uint64_t> : Vec512iTraitsBase {
        template <bool IsMin>
        static Type MinMax(Type a, Type b) noexcept {
            return IsMin ? _mm512_min_epu64(a, b) : _mm512_max_epu64(a, b);
        }

        template <bool IsMin>
        static uint64_t ReduceMin(Type a, Type b) noexcept {
            return IsMin ? _mm512_reduce_min_epu64(a) : _mm512_reduce_max_epu64(a);
        }
    };
#endif

    template <MinMaxMode Mode, class Traits, typename T>
    auto GetResult(auto v_min, auto v_max) noexcept {
        T min_val;
        if constexpr (Mode & ModeMin) {
            min_val = std::bit_cast<T>(Traits::template ReduceMinMax<true>(v_min));
        }
        if constexpr (Mode == ModeMin) {
            return min_val;
        } else {
            T max_val;
            max_val = std::bit_cast<T>(Traits::template ReduceMinMax<false>(v_max));
            if constexpr (Mode == ModeMax) {
                return max_val;
            } else {
                return std::pair{min_val, max_val};
            }
        }
    }

    template <MinMaxMode Mode, class Traits, typename T>
    auto Impl1xVec(const T* data) noexcept {
        const auto v = Traits::Load(data);
        return GetResult<Mode, Traits, T>(v, v);
    }

    template <MinMaxMode Mode, class Traits, typename T>
    auto ImplGt1Lt2xVec(const T* begin, const T* end) noexcept {
        using VecType = typename Traits::Type;
        const auto v1 = Traits::Load(begin);
        const auto v2 = Traits::Load(reinterpret_cast<const VecType*>(end) - 1);
        return GetResult<Mode, Traits, T>(Traits::template MinMax<true>(v1, v2), Traits::template MinMax<false>(v1, v2));
    }

    template <MinMaxMode Mode, class Traits, typename T>
    auto ImplGt1xVec(const T* begin, const T* end) noexcept {
        using VecType = typename Traits::Type;

        const auto* first = reinterpret_cast<const VecType*>(begin);
        const auto* last = reinterpret_cast<const VecType*>(end) - 1;

        const auto v1 = Traits::Load(first);
        const auto v2 = Traits::Load(last);
        ++first;

        VecType v_min, v_max;
        if constexpr (Mode & ModeMin) {
            v_min = Traits::template MinMax<true>(v1, v2);
        }
        if constexpr (Mode & ModeMax) {
            v_max = Traits::template MinMax<false>(v1, v2);
        }

        for (; first < last; ++first) {
            const auto v = Traits::Load(first);
            if constexpr (Mode & ModeMin) {
                v_min = Traits::template MinMax<true>(v_min, v);
            }
            if constexpr (Mode & ModeMax) {
                v_max = Traits::template MinMax<false>(v_max, v);
            }
        }

        return GetResult<Mode, Traits, T>(v_min, v_max);
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


    template <typename T, typename Elem = std::remove_cvref_t<T>>
    concept CanVectorize = !std::is_volatile_v<std::remove_reference_t<T>> && (std::floating_point<Elem> && sizeof(Elem) <= sizeof(double) || std::integral<Elem> || std::is_pointer_v<Elem>);

    template <CanVectorize T>
    constexpr auto ToArithmeticType() noexcept {
        constexpr bool is_signed = std::is_signed_v<T>;

        if constexpr (std::floating_point<std::remove_cvref_t<T>>) {
            return std::remove_cvref_t<T>{};
        } else if constexpr (sizeof(T) == 1) {
            if constexpr (is_signed) {
                return int8_t{};
            } else {
                return uint8_t{};
            }
        } else if constexpr (sizeof(T) == 2) {
            if constexpr (is_signed) {
                return int16_t{};
            } else {
                return uint16_t{};
            }
        } else if constexpr (sizeof(T) == 4) {
            if constexpr (is_signed) {
                return int32_t{};
            } else {
                return uint32_t{};
            }
        } else {
            if constexpr (is_signed) {
                return int64_t{};
            } else {
                return uint64_t{};
            }
        }
    }

    template <MinMaxMode Mode, typename T>
    auto Impl(const T* data, size_t len) noexcept {
        const T* end = data + len;
        using ElemType = decltype(ToArithmeticType<T>());

        constexpr int VecBits{
#if defined(__AVX512BW__)
                512
#elif defined(__AVX512F__)
                std::integral<ElemType> && sizeof(ElemType) <= 2 ? 256 : 512
#elif defined(__AVX2__)
                256
#elif defined(__AVX__)
                std::integral<ElemType> ? 128 : 256
#elif defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
                128
#endif
        };

        constexpr size_t VecElems = VecBits / 8 / sizeof(ElemType);

        if constexpr (VecBits >= 128) {
            if (len >= VecElems) {
                using Traits = VecTraits<VecBits, ElemType>;
                if (len == VecElems) {
                    return Impl1xVec<Mode, Traits>(data);
                }
                return ImplGt1xVec<Mode, Traits>(data, end);
            }
        }
        if constexpr (VecBits >= 256) {
            if (len >= VecElems / 2) {
                using Traits = VecTraits<VecBits / 2, ElemType>;
                if (len == VecElems / 2) {
                    return Impl1xVec<Mode, Traits>(data);
                }
                return ImplGt1Lt2xVec<Mode, Traits>(data, end);
            }
        }
        if constexpr (VecBits == 512) {
            if (len >= VecElems / 4) {
                using Traits = VecTraits<VecBits / 4, ElemType>;
                if (len == VecElems / 4) {
                    return Impl1xVec<Mode, Traits>(data);
                }
                return ImplGt1Lt2xVec<Mode, Traits>(data, end);
            }
        }
        return ScalarImpl<Mode>(data, end);
    }
}  // namespace detail


template <detail::CanVectorize T>
T Min(const T* data, size_t len) noexcept {
    return detail::Impl<detail::ModeMin>(data, len);
}

template <detail::CanVectorize T>
T Max(const T* data, size_t len) noexcept {
    return detail::Impl<detail::ModeMax>(data, len);
}

template <detail::CanVectorize T>
std::pair<T, T> MinMax(const T* data, size_t len) noexcept {
    return detail::Impl<detail::ModeMinMax>(data, len);
}
