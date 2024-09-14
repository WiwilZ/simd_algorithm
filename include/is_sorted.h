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
#include <functional>
#include <utility>


namespace detail {
    template <typename T>
    T GenMask(size_t len) noexcept {
#ifdef __BMI2__
        if constexpr (sizeof(T) <= 4) {
            return _bzhi_u32(-1, len);
        } else {
            return _bzhi_u64(-1, len);
        }
#else
        if constexpr (sizeof(T) <= 4) {
            return ~(-1 << len);
        } else {
            return ~(-1ll << len);
        }
#endif
    }

    template <typename T>
    T ClipMask(T mask, size_t len) {
#ifdef __BMI2__
        if constexpr (sizeof(T) <= 4) {
            return _bzhi_u32(mask, len);
        } else {
            return _bzhi_u64(mask, len);
        }
#else
        return mask & GenMask<T>(len);
#endif
    }

    template <size_t VecBits, typename ElemType>
    struct VecTraits;

#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)


    template <typename ElemType>
    struct Vec128fTraits {
        using Type = std::conditional_t<sizeof(ElemType) == 4, __m128, __m128d>;

#if defined(__SSE4_1__) || defined(__AVX__)
        static bool TestAllZeros(Type a) noexcept {
#if defined(__AVX__)
            return sizeof(ElemType) == 4 ? _mm_testz_ps(a, a) : _mm_testz_pd(a, a);
#else
            const auto v = sizeof(ElemType) == 4 ? _mm_castsi128_ps(a) : _mm_castsi128_pd(a);
            return _mm_testz_si128(v, v);
#endif
        }
#endif

        static int MoveMask(Type a) noexcept {
            return sizeof(ElemType) == 4 ? _mm_movemask_ps(a) : _mm_movemask_pd(a);
        }

        static Type LessThan(Type a, Type b) noexcept {
            return sizeof(ElemType) == 4 ? _mm_cmplt_ps(a, b) : _mm_cmplt_pd(a, b);
        }

        static Type LessEqual(Type a, Type b) noexcept {
            return sizeof(ElemType) == 4 ? _mm_cmple_ps(a, b) : _mm_cmple_pd(a, b);
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(LessThan(a, b));
#else
            return MoveMask(LessThan(a, b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(LessEqual(a, b));
#else
            return MoveMask(LessThan(a, b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return ClipMask(MoveMask(LessThan(a, b)), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return ClipMask(MoveMask(LessThan(a, b)), len);
        }
    };

    template <>
    struct VecTraits<128, float> : Vec128fTraits<float> {
        static Type Load(const void* data) noexcept {
            return _mm_loadu_ps(static_cast<const float*>(data));
        }

#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmp_ps_mask(a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_ps_mask(0xf, _mm512_castps128_ps512(a), _mm512_castps128_ps512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmp_ps_mask(a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_ps_mask(0xf, _mm512_castps128_ps512(a), _mm512_castps128_ps512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmp_ps_mask(mask, a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_ps_mask(mask, _mm512_castps128_ps512(a), _mm512_castps128_ps512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmp_ps_mask(mask, a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_ps_mask(mask, _mm512_castps128_ps512(a), _mm512_castps128_ps512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, double> : Vec128fTraits<double> {
        static Type Load(const void* data) noexcept {
            return _mm_loadu_pd(static_cast<const double*>(data));
        }

#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmp_pd_mask(a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_pd_mask(3, _mm512_castpd128_pd512(a), _mm512_castpd128_pd512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmp_pd_mask(a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_pd_mask(3, _mm512_castpd128_pd512(a), _mm512_castpd128_pd512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmp_pd_mask(mask, a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_pd_mask(mask, _mm512_castpd128_pd512(a), _mm512_castpd128_pd512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmp_pd_mask(mask, a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_pd_mask(mask, _mm512_castpd128_pd512(a), _mm512_castpd128_pd512(b));
#endif
        }
#endif
    };


    template <typename ElemType>
    struct Vec128iTraitsBase {
        using Type = __m128i;

        static Type Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const Type*>(data));
        }

#if defined(__SSE4_1__) || defined(__AVX__)
        static bool TestAllZeros(Type a) noexcept {
            return _mm_test_all_zeros(a, a);
        }

        static bool TestAllOnes(Type a) noexcept {
            return _mm_test_all_ones(a);
        }
#endif

        static int MoveMask(Type a) noexcept {
            if constexpr (sizeof(ElemType) == 8) {
                return _mm_movemask_pd(_mm_castsi128_pd(a));
            } else if constexpr (sizeof(ElemType) == 4) {
                return _mm_movemask_ps(_mm_castsi128_ps(a));
            } else {
                return _mm_movemask_epi8(a);
            }
        }
    };

    template <typename ElemType>
    struct Vec128iTraitsBase1 : Vec128iTraitsBase<ElemType> {
        using Type = Vec128iTraitsBase<ElemType>::Type;
        using Vec128iTraitsBase<ElemType>::TestAllZeros;
        using Vec128iTraitsBase<ElemType>::MoveMask;

        static Type GreaterThan(Type a, Type b) noexcept {
            if constexpr (sizeof(ElemType) == 1) {
                return _mm_cmpgt_epi8(a, b);
            } else if constexpr (sizeof(ElemType) == 2) {
                return _mm_cmpgt_epi16(a, b);
            } else if constexpr (sizeof(ElemType) == 4) {
                return _mm_cmpgt_epi32(a, b);
            } else {
#if defined(__SSE4_2__) || defined(__AVX__)
                return _mm_cmpgt_epi64(a, b);
#else
                static const auto offset = _mm_set1_epi64x(0x8000'0000);
                a = _mm_xor_si128(a, offset);
                b = _mm_xor_si128(b, offset);
                // a[1, 3] > b[1, 3] || (a[1, 3] == b[1, 3] && a[0, 2] > b[0, 2])
                const auto gt = _mm_cmpgt_epi32(a, b);
                const auto eq = _mm_cmpeq_epi32(a, b);
                const auto gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
                return _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
            }
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
            constexpr int level{
#if defined(__SSE4_2__) || defined(__AVX__)
                    2
#elif defined(__SSE4_1__)
                    1
#endif
            };
            if constexpr (sizeof(ElemType) < 8 && level > 0 || sizeof(ElemType) == 8 && level == 2) {
                return !TestAllZeros(GreaterThan(b, a));
            } else {
                return MoveMask(GreaterThan(b, a));
            }
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            constexpr int mask = sizeof(ElemType) <= 2 ? 0xffff : sizeof(ElemType) == 4 ? 0xf
                                                                                        : 3;
            return MoveMask(GreaterThan(a, b)) != mask;
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(MoveMask(GreaterThan(b, a)), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(~MoveMask(GreaterThan(a, b)), len);
        }
    };

    template <>
    struct VecTraits<128, int8_t> : Vec128iTraitsBase1<int8_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmplt_epi8_mask(a, b);
#else
            return _mm512_mask_cmplt_epi8_mask(0xffff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmple_epi8_mask(a, b);
#else
            return _mm512_mask_cmple_epi8_mask(0xffff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epi8_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi8_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epi8_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi8_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint8_t> : Vec128iTraitsBase<uint8_t> {
        static Type Min(Type a, Type b) noexcept {
            return _mm_min_epu8(a, b);
        }

        static Type Max(Type a, Type b) noexcept {
            return _mm_max_epu8(a, b);
        }

        static Type Equal(Type a, Type b) noexcept {
            return _mm_cmpeq_epi8(a, b);
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_cmplt_epu8_mask(a, b);
#elif defined(__AVX512BW__)
            return _mm512_mask_cmplt_epu8_mask(0xffff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(_mm_xor_si128(Max(a, b), a));
#else
            return MoveMask(Equal(Max(a, b), a)) != 0xffff;
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_cmple_epu8_mask(a, b);
#elif defined(__AVX512BW__)
            return _mm512_mask_cmple_epu8_mask(0xffff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(Equal(Min(a, b), a));
#else
            return MoveMask(Equal(Min(a, b), a));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512BW__
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epu8_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu8_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
            return ClipMask(~MoveMask(Equal(Max(a, b), a)), len);
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512BW__
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epu8_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu8_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
            return ClipMask(MoveMask(Equal(Min(a, b), a)), len);
#endif
        }
    };

    template <>
    struct VecTraits<128, int16_t> : Vec128iTraitsBase1<int16_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmplt_epi16_mask(a, b);
#else
            return _mm512_mask_cmplt_epi16_mask(0xff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmple_epi16_mask(a, b);
#else
            return _mm512_mask_cmple_epi16_mask(0xff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epi16_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi16_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epi16_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi16_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint16_t> : Vec128iTraitsBase<uint16_t> {
#if defined(__SSE4_1__) || defined(__AVX__)
        static Type Min(Type a, Type b) noexcept {
            return _mm_min_epu16(a, b);
        }

        static Type Max(Type a, Type b) noexcept {
            return _mm_max_epu16(a, b);
        }
#endif

        static Type Equal(Type a, Type b) noexcept {
            return _mm_cmpeq_epi16(a, b);
        }

        static Type LessEqual(Type a, Type b) noexcept {
            return Equal(_mm_subs_epu16(a, b), _mm_setzero_si128());
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_cmplt_epu16_mask(a, b);
#elif defined(__AVX512BW__)
            return _mm512_mask_cmplt_epu16_mask(0xff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(_mm_xor_si128(Max(a, b), a));
#else
            return MoveMask(LessEqual(b, a)) != 0xffff;
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            return _mm_cmple_epu16_mask(a, b);
#elif defined(__AVX512BW__)
            return _mm512_mask_cmple_epu16_mask(0xff, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(Equal(Min(a, b), a));
#else
            return MoveMask(LessEqual(a, b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512BW__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epu16_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu16_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
#if defined(__SSE4_1__) || defined(__AVX__)
            const auto v = Equal(Max(a, b), a);
#else
            const auto v = LessEqual(b, a);
#endif
            return ClipMask(~MoveMask(v), len << 1);

#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512BW__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epu16_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu16_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
#if defined(__SSE4_1__) || defined(__AVX__)
            const auto v = Equal(Min(a, b), a);
#else
            const auto v = LessEqual(a, b);
#endif
            return ClipMask(MoveMask(v), len << 1);
#endif
        }
    };

    template <>
    struct VecTraits<128, int32_t> : Vec128iTraitsBase1<int32_t> {
#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmplt_epi32_mask(a, b);
#else
            return _mm512_mask_cmplt_epi32_mask(0xf, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmple_epi32_mask(a, b);
#else
            return _mm512_mask_cmple_epi32_mask(0xf, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epi32_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi32_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epi32_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi32_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint32_t> : Vec128iTraitsBase<uint32_t> {
#if defined(__SSE4_1__) || defined(__AVX__)
        static Type Min(Type a, Type b) noexcept {
            return _mm_min_epu32(a, b);
        }

        static Type Max(Type a, Type b) noexcept {
            return _mm_max_epu32(a, b);
        }
#endif

        static Type Equal(Type a, Type b) noexcept {
            return _mm_cmpeq_epi32(a, b);
        }

        static Type GreaterThan(Type a, Type b) noexcept {
            static const auto offset = _mm_set1_epi32(0x8000'0000);
            return _mm_cmpgt_epi32(_mm_xor_si128(a, offset), _mm_xor_si128(b, offset));
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm_cmplt_epu32_mask(a, b);
#elif defined(__AVX512F__)
            return _mm512_mask_cmplt_epu32_mask(0xf, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(_mm_xor_si128(Max(a, b), a));
#else
            return MoveMask(GreaterThan(b, a));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm_cmple_epu32_mask(a, b);
#elif defined(__AVX512F__)
            return _mm512_mask_cmple_epu32_mask(0xf, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_1__) || defined(__AVX__)
            return !TestAllZeros(Equal(Min(a, b), a));
#else
            return MoveMask(GreaterThan(a, b)) != 0xf;
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epu32_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu32_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
#if defined(__SSE4_1__) || defined(__AVX__)
            const auto mask = ~MoveMask(Equal(Max(a, b), a));
#else
            const auto mask = MoveMask(GreaterThan(b, a));
#endif
            return ClipMask(mask, len);
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epu32_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu32_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
#if defined(__SSE4_1__) || defined(__AVX__)
            const auto mask = MoveMask(Equal(Min(a, b), a));
#else
            const auto mask = ~MoveMask(GreaterThan(a, b));
#endif
            return ClipMask(mask, len);
#endif
        }
    };

    template <>
    struct VecTraits<128, int64_t> : Vec128iTraitsBase1<int64_t> {
#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmplt_epi64_mask(a, b);
#else
            return _mm512_mask_cmplt_epi64_mask(3, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm_cmple_epi64_mask(a, b);
#else
            return _mm512_mask_cmple_epi64_mask(3, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epi64_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi64_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epi64_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi64_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<128, uint64_t> : Vec128iTraitsBase<uint64_t> {
        static Type GreaterThan(Type a, Type b) noexcept {
#if defined(__SSE4_2__) || defined(__AVX__)
            static const auto offset = _mm_set1_epi64x(0x8000'0000'0000'0000);
            return _mm_cmpgt_epi64(_mm_xor_si128(a, offset), _mm_xor_si128(b, offset));
#else
            static const auto offset = _mm_set1_epi64x(0x8000'0000'8000'0000);
            a = _mm_xor_si128(a, offset);
            b = _mm_xor_si128(b, offset);
            // a[1, 3] > b[1, 3] || (a[1, 3] == b[1, 3] && a[0, 2] > b[0, 2])
            const auto gt = _mm_cmpgt_epi32(a, b);
            const auto eq = _mm_cmpeq_epi32(a, b);
            const auto gt_low = _mm_shuffle_epi32(gt, _MM_SHUFFLE(2, 2, 0, 0));
            return _mm_or_si128(gt, _mm_and_si128(eq, gt_low));
#endif
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm_cmplt_epu64_mask(a, b);
#elif defined(__AVX512F__)
            return _mm512_mask_cmplt_epu64_mask(3, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#elif defined(__SSE4_2__) || defined(__AVX__)
            return !TestAllZeros(GreaterThan(b, a));
#else
            return MoveMask(GreaterThan(b, a));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm_cmple_epu64_mask(a, b);
#elif defined(__AVX512F__)
            return _mm512_mask_cmple_epu64_mask(3, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#else
            return MoveMask(GreaterThan(a, b)) != 3;
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmplt_epu64_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu64_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
            return ClipMask(MoveMask(GreaterThan(b, a)), len);
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm_mask_cmple_epu64_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu64_mask(mask, _mm512_castsi128_si512(a), _mm512_castsi128_si512(b));
#endif
#else
            return ClipMask(~MoveMask(GreaterThan(a, b)), len);
#endif
        }
    };
#endif  // defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)


#ifdef __AVX__
    template <>
    struct VecTraits<256, float> {
        using Type = __m256;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_ps(static_cast<const float*>(data));
        }

        static bool TestAllZeros(Type a) noexcept {
            return _mm256_testz_ps(a, a);
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
#elif defined(__AVX512F__)
            return _mm512_mask_cmplt_ps_mask(0xff, _mm512_castps256_ps512(a), _mm512_castps256_ps512(b));
#else
            return !TestAllZeros(_mm256_cmp_ps(a, b, _CMP_LT_OS));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm256_cmp_ps_mask(a, b, _CMP_LE_OS);
#elif defined(__AVX512F__)
            return _mm512_mask_cmple_ps_mask(0xff, _mm512_castps256_ps512(a), _mm512_castps256_ps512(b));
#else
            return !TestAllZeros(_mm256_cmp_ps(a, b, _CMP_LE_OS));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmp_ps_mask(mask, a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_ps_mask(mask, _mm512_castps256_ps512(a), _mm512_castps256_ps512(b));
#endif
#else
            return ClipMask(_mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_LT_OS)), len);
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmp_ps_mask(mask, a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_ps_mask(mask, _mm512_castps256_ps512(a), _mm512_castps256_ps512(b));
#endif
#else
            return ClipMask(_mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_LE_OS)), len);
#endif
        }
    };

    template <>
    struct VecTraits<256, double> {
        using Type = __m256d;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_pd(static_cast<const double*>(data));
        }

        static bool TestAllZeros(Type a) noexcept {
            return _mm256_testz_pd(a, a);
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm256_cmp_pd_mask(a, b, _CMP_LT_OS);
#elif defined(__AVX512F__)
            return _mm512_mask_cmplt_pd_mask(0xf, _mm512_castpd256_pd512(a), _mm512_castpd256_pd512(b));
#else
            return !TestAllZeros(_mm256_cmp_pd(a, b, _CMP_LT_OS));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#if defined(__AVX512VL__)
            return _mm256_cmp_pd_mask(a, b, _CMP_LE_OS);
#elif defined(__AVX512F__)
            return _mm512_mask_cmple_pd_mask(0xf, _mm512_castpd256_pd512(a), _mm512_castpd256_pd512(b));
#else
            return !TestAllZeros(_mm256_cmp_pd(a, b, _CMP_LE_OS));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmp_pd_mask(mask, a, b, _CMP_LT_OS);
#else
            return _mm512_mask_cmplt_pd_mask(mask, _mm512_castpd256_pd512(a), _mm512_castpd256_pd512(b));
#endif
#else
            return ClipMask(_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LT_OS)), len);
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
#ifdef __AVX512F__
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmp_pd_mask(mask, a, b, _CMP_LE_OS);
#else
            return _mm512_mask_cmple_pd_mask(mask, _mm512_castpd256_pd512(a), _mm512_castpd256_pd512(b));
#endif
#else
            return ClipMask(_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LE_OS)), len);
#endif
        }
    };

    template <typename ElemType>
    struct Vec256iTraitsBase {
        using Type = __m256i;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_si256(static_cast<const __m256i*>(data));
        }

        static bool TestAllZeros(Type a) noexcept {
            return _mm256_testz_si256(a, a);
        }

        static bool TestAllOnes(Type a) noexcept {
            const auto ones = _mm256_cmpeq_epi32(a, a);
            return _mm256_testc_si256(a, ones);
        }

        static int MoveMask(Type a) noexcept {
            if constexpr (sizeof(ElemType) == 8) {
                return _mm256_movemask_pd(_mm256_castsi256_pd(a));
            } else if constexpr (sizeof(ElemType) == 4) {
                return _mm256_movemask_ps(_mm256_castsi256_ps(a));
            } else {
#ifdef __AVX2__
                return _mm256_movemask_epi8(a);
#endif
            }
        }

        static auto Apart(Type a, Type b, auto func) noexcept {
            const auto a_low = _mm256_castsi256_si128(a);
            const auto b_low = _mm256_castsi256_si128(b);
            const auto a_high = _mm256_extractf128_si256(a, 1);
            const auto b_high = _mm256_extractf128_si256(b, 1);
            const auto low = func(a_low, b_low);
            const auto high = func(a_high, b_high);
            return std::pair{low, high};
        };
    };

    template <typename ElemType>
    struct Vec256iTraitsBase1 : Vec256iTraitsBase<ElemType> {
        using Type = Vec256iTraitsBase<ElemType>::Type;
        using Vec256iTraitsBase<ElemType>::TestAllZeros;
        using Vec256iTraitsBase<ElemType>::MoveMask;

#ifdef __AVX2__
        static Type GreaterThan(Type a, Type b) noexcept {
            if constexpr (sizeof(ElemType) == 1) {
                return _mm256_cmpgt_epi8(a, b);
            } else if constexpr (sizeof(ElemType) == 2) {
                return _mm256_cmpgt_epi16(a, b);
            } else if constexpr (sizeof(ElemType) == 4) {
                return _mm256_cmpgt_epi32(a, b);
            } else {
                return _mm256_cmpgt_epi64(a, b);
            }
        }
#endif

        static auto ApartGreaterThan(Type a, Type b) noexcept {
            return Apart(a, b, GreaterThan);
        };

        static int GreaterThanMask(Type a, Type b) noexcept {
#ifdef __AVX2__
            return MoveMask(GreaterThan(a, b));
#else
            const auto [low, high] = ApartGreaterThan(a, b);
            if constexpr (sizeof(ElemType) >= 4) {
                return MoveMask(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
            } else {
                const int low_mask = VecTraits<128, ElemType>::MoveMask(low);
                const int high_mask = VecTraits<128, ElemType>::MoveMask(high);
                return high_mask << 16 | low_mask;
            }
#endif
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX2__
            return !TestAllZeros(GreaterThan(b, a));
#else
            const auto [low, high] = ApartGreaterThan(b, a);
            return !VecTraits<128, ElemType>::TestAllZeros(_mm_or_si128(low, high));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX2__
            constexpr int mask = sizeof(ElemType) <= 2 ? 0xffffffff : sizeof(ElemType) == 4 ? 0xff
                                                                                            : 0xf;
            return MoveMask(GreaterThan(a, b)) != mask;
#else
            constexpr int mask = sizeof(ElemType) <= 2 ? 0xffff : sizeof(ElemType) == 4 ? 0xf
                                                                                        : 3;
            const auto [low, high] = ApartGreaterThan(a, b);
            return VecTraits<128, ElemType>::MoveMask(_mm_and_si128(low, high)) != mask;
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(GreaterThanMask(b, a), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(~GreaterThanMask(a, b), len);
        }
    };

    template <typename ElemType>
    struct Vec256iTraitsBase2 : Vec256iTraitsBase<ElemType> {
        using Type = Vec256iTraitsBase<ElemType>::Type;
        using Vec256iTraitsBase<ElemType>::TestAllZeros;
        using Vec256iTraitsBase<ElemType>::MoveMask;

#ifdef __AVX2__
        static Type Min(Type a, Type b) noexcept {
            if constexpr (sizeof(ElemType) == 1) {
                return _mm256_min_epu8(a, b);
            } else if constexpr (sizeof(ElemType) == 2) {
                return _mm256_min_epu16(a, b);
            } else {
                return _mm256_min_epu32(a, b);
            }
        }

        static Type Max(Type a, Type b) noexcept {
            if constexpr (sizeof(ElemType) == 1) {
                return _mm256_max_epu8(a, b);
            } else if constexpr (sizeof(ElemType) == 2) {
                return _mm256_max_epu16(a, b);
            } else {
                return _mm256_max_epu32(a, b);
            }
        }

        static Type Equal(Type a, Type b) noexcept {
            if constexpr (sizeof(ElemType) == 1) {
                return _mm256_cmpeq_epi8(a, b);
            } else if constexpr (sizeof(ElemType) == 2) {
                return _mm256_cmpeq_epi16(a, b);
            } else {
                return _mm256_cmpeq_epi32(a, b);
            }
        }
#endif

        template <bool IsMin>
        static int CmpMask(Type a, Type b) noexcept {
#ifdef __AVX2__
            return MoveMask(Equal((IsMin ? Min : Max)(a, b), a));
#else
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return VecTraits<128, ElemType>::Equal((IsMin ? VecTraits<128, ElemType>::Min : VecTraits<128, ElemType>::Max)(a, b), a); });
            const int low_mask = VecTraits<128, ElemType>::MoveMask(low);
            const int high_mask = VecTraits<128, ElemType>::MoveMask(high);
            return high_mask << 16 | low_mask;
#endif
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX2__
            return !TestAllZeros(_mm256_xor_si256(Max(a, b), a));
#else
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return _mm_xor_si128(VecTraits<128, ElemType>::Max(a, b), a); });
            return !VecTraits<128, ElemType>::TestAllZeros(_mm_or_si128(low, high));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX2__
            !TestAllZeros(Equal(Min(a, b), a));
#else
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return VecTraits<128, ElemType>::Equal(VecTraits<128, ElemType>::Min(a, b), a); });
            return !VecTraits<128, ElemType>::TestAllZeros(_mm_or_si128(low, high));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(~CmpMask<false>(a, b), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(CmpMask<true>(a, b), len);
        }
    };

    template <>
    struct VecTraits<256, int8_t> : Vec256iTraitsBase1<int8_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epi8_mask(a, b);
#else
            return _mm512_mask_cmplt_epi8_mask(0xffffffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epi8_mask(a, b);
#else
            return _mm512_mask_cmple_epi8_mask(0xffffffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask32>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epi8_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi8_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask32>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epi8_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi8_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, uint8_t> : Vec256iTraitsBase2<uint8_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epu8_mask(a, b);
#else
            return _mm512_mask_cmplt_epu8_mask(0xffffffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epu8_mask(a, b);
#else
            return _mm512_mask_cmple_epu8_mask(0xffffffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask32>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epu8_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu8_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask32>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epu8_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu8_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, int16_t> : Vec256iTraitsBase1<int16_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epi16_mask(a, b);
#else
            return _mm512_mask_cmplt_epi16_mask(0xffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epi16_mask(a, b);
#else
            return _mm512_mask_cmple_epi16_mask(0xffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epi16_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi16_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epi16_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi16_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, uint16_t> : Vec256iTraitsBase2<uint16_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epu16_mask(a, b);
#else
            return _mm512_mask_cmplt_epu16_mask(0xffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epu16_mask(a, b);
#else
            return _mm512_mask_cmple_epu16_mask(0xffff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epu16_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu16_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask16>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epu16_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu16_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, int32_t> : Vec256iTraitsBase1<int32_t> {
#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epi32_mask(a, b);
#else
            return _mm512_mask_cmplt_epi32_mask(0xff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epi32_mask(a, b);
#else
            return _mm512_mask_cmple_epi32_mask(0xff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epi32_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi32_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epi32_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi32_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, uint32_t> : Vec256iTraitsBase2<uint32_t> {
#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epu32_mask(a, b);
#else
            return _mm512_mask_cmplt_epu32_mask(0xff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epu32_mask(a, b);
#else
            return _mm512_mask_cmple_epu32_mask(0xff, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epu32_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu32_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epu32_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu32_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, int64_t> : Vec256iTraitsBase1<int64_t> {
#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epi64_mask(a, b);
#else
            return _mm512_mask_cmplt_epi64_mask(0xf, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epi64_mask(a, b);
#else
            return _mm512_mask_cmple_epi64_mask(0xf, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epi64_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epi64_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epi64_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epi64_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };

    template <>
    struct VecTraits<256, uint64_t> : Vec256iTraitsBase1<uint64_t> {
        static inline const auto offset = _mm256_set1_epi64x(0x8000'0000'0000'0000);

#ifdef __AVX2__
        static Type GreaterThan(Type a, Type b) noexcept {
            return _mm256_cmpgt_epi64(_mm256_xor_si256(a, offset), _mm256_xor_si256(b, offset));
        }
#endif

        static auto ApartGreaterThan(Type a, Type b) noexcept {
            a = _mm256_xor_si256(a, offset);
            b = _mm256_xor_si256(b, offset);
            return Apart(a, b, VecTraits<128, int64_t>::GreaterThan);
        };

#ifdef __AVX512F__
        static bool AnyLessThan(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmplt_epu64_mask(a, b);
#else
            return _mm512_mask_cmplt_epu64_mask(0xf, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
#ifdef __AVX512VL__
            return _mm256_cmple_epu64_mask(a, b);
#else
            return _mm512_mask_cmple_epu64_mask(0xf, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmplt_epu64_mask(mask, a, b);
#else
            return _mm512_mask_cmplt_epu64_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const auto mask = GenMask<__mmask8>(len);
#ifdef __AVX512VL__
            return _mm256_mask_cmple_epu64_mask(mask, a, b);
#else
            return _mm512_mask_cmple_epu64_mask(mask, _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
#endif
        }
#endif
    };
#endif  // __AVX__


#ifdef __AVX512F__
    template <>
    struct VecTraits<512, float> {
        using Type = __m512;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_ps(data);
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_ps_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_ps_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_ps_mask(GenMask<__mmask16>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_ps_mask(GenMask<__mmask16>(len), a, b);
        }
    };

    template <>
    struct VecTraits<512, double> {
        using Type = __m512d;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_pd(data);
        }
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_pd_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_pd_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_pd_mask(GenMask<__mmask8>(len), a, b);
        }
        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_pd_mask(GenMask<__mmask8>(len), a, b);
        }
    };


    struct Vec512iTraitsBase {
        using Type = __m512i;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }

        static auto Apart(Type a, Type b, auto func) noexcept {
            const auto a_low = _mm512_castsi512_si256(a);
            const auto b_low = _mm512_castsi512_si256(b);
            const auto a_high = _mm512_extracti64x4_epi64(a, 1);
            const auto b_high = _mm512_extracti64x4_epi64(b, 1);
            const auto low = func(a_low, b_low);
            const auto high = func(a_high, b_high);
            return std::pair{low, high};
        }
    };

    template <typename ElemType>
    struct Vec512iTraitsBase1 : Vec512iTraitsBase {
        static auto ApartGreaterThan(Type a, Type b) noexcept {
            return Apart(a, b, VecTraits<256, ElemType>::GreaterThan);
        };

        static __mmask64 GreaterThanMask(Type a, Type b) noexcept {
            const auto [low, high] = ApartGreaterThan(a, b);
            const __mmask64 low_mask = VecTraits<256, ElemType>::MoveMask(low);
            const __mmask64 high_mask = VecTraits<256, ElemType>::MoveMask(high);
            return high_mask << 32 | low_mask;
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
            const auto [low, high] = ApartGreaterThan(b, a);
            return !VecTraits<256, ElemType>::TestAllZeros(_mm256_or_si256(low, high));
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            const auto [low, high] = ApartGreaterThan(a, b);
            return VecTraits<256, ElemType>::MoveMask(_mm256_and_si256(low, high)) != 0xffffffff;
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(GreaterThanMask(b, a), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(~GreaterThanMask(a, b), len);
        }
    };

    template <typename ElemType>
    struct Vec512iTraitsBase2 : Vec512iTraitsBase {
        template <bool IsMin>
        static __mmask64 CmpMask(Type a, Type b) noexcept {
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return VecTraits<256, ElemType>::Equal((IsMin ? VecTraits<256, ElemType>::Min : VecTraits<256, ElemType>::Max)(a, b), a); });
            const __mmask64 low_mask = VecTraits<256, ElemType>::MoveMask(low);
            const __mmask64 high_mask = VecTraits<256, ElemType>::MoveMask(high);
            return high_mask << 32 | low_mask;
        }

        static bool AnyLessThan(Type a, Type b) noexcept {
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return _mm256_xor_si256(VecTraits<256, ElemType>::Max(a, b), a); });
            return !VecTraits<256, ElemType>::TestAllZeros(_mm256_or_si256(low, high));
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            const auto [low, high] = Apart(a, b, [](auto a, auto b) { return VecTraits<256, ElemType>::Equal(VecTraits<256, ElemType>::Min(a, b), a); });
            return !VecTraits<256, ElemType>::TestAllZeros(_mm256_or_si256(low, high));
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(~CmpMask<false>(a, b), len);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            if constexpr (sizeof(ElemType) == 2) {
                len <<= 1;
            }
            return ClipMask(CmpMask<true>(a, b), len);
        }
    };

    template <>
    struct VecTraits<512, int8_t> : Vec512iTraitsBase1<int8_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epi8_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epi8_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epi8_mask(GenMask<__mmask64>(len), _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epi8_mask(GenMask<__mmask64>(len), _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
        }
#endif
    };

    template <>
    struct VecTraits<512, uint8_t> : Vec512iTraitsBase2<uint8_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epu8_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epu8_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epu8_mask(GenMask<__mmask64>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epu8_mask(GenMask<__mmask64>(len), a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int16_t> : Vec512iTraitsBase1<int16_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epi16_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epi16_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epi16_mask(GenMask<__mmask32>(len), _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epi16_mask(GenMask<__mmask32>(len), _mm512_castsi256_si512(a), _mm512_castsi256_si512(b));
        }
#endif
    };

    template <>
    struct VecTraits<512, uint16_t> : Vec512iTraitsBase2<uint16_t> {
#ifdef __AVX512BW__
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epu16_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epu16_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epu16_mask(GenMask<__mmask32>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epu16_mask(GenMask<__mmask32>(len), a, b);
        }
#endif
    };

    template <>
    struct VecTraits<512, int32_t> : Vec512iTraitsBase {
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epi32_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epi32_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epi32_mask(GenMask<__mmask16>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epi32_mask(GenMask<__mmask16>(len), a, b);
        }
    };

    template <>
    struct VecTraits<512, uint32_t> : Vec512iTraitsBase {
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epu32_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epu32_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epu32_mask(GenMask<__mmask16>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epu32_mask(GenMask<__mmask16>(len), a, b);
        }
    };

    template <>
    struct VecTraits<512, int64_t> : Vec512iTraitsBase {
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epi64_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epi64_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epi64_mask(GenMask<__mmask8>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            const __mmask8 mask = ~(-1ll << len);
            return _mm512_mask_cmple_epi64_mask(GenMask<__mmask8>(len), a, b);
        }
    };

    template <>
    struct VecTraits<512, uint64_t> : Vec512iTraitsBase {
        static bool AnyLessThan(Type a, Type b) noexcept {
            return _mm512_cmplt_epu64_mask(a, b);
        }

        static bool AnyLessEqual(Type a, Type b) noexcept {
            return _mm512_cmple_epu64_mask(a, b);
        }

        static bool AnyLessThan(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmplt_epu64_mask(GenMask<__mmask8>(len), a, b);
        }

        static bool AnyLessEqual(Type a, Type b, size_t len) noexcept {
            return _mm512_mask_cmple_epu64_mask(GenMask<__mmask8>(len), a, b);
        }
    };
#endif  // __AVX512F__


    enum class SortMode {
        Increasing,
        StrictIncreasing,
        Decreasing,
        StrictDecreasing
    };

    template <SortMode Mode>
    constexpr auto ToCompareType() noexcept {
        if constexpr (Mode == SortMode::Increasing) {
            return std::less<>{};
        } else if constexpr (Mode == SortMode::StrictIncreasing) {
            return std::less_equal<>{};
        } else if constexpr (Mode == SortMode::Decreasing) {
            return std::greater<>{};
        } else {
            return std::greater_equal<>{};
        }
    }

    template <class Traits, SortMode Mode>
    static bool Compare(auto next, auto curr) noexcept {
        if constexpr (Mode == SortMode::Increasing) {
            return Traits::AnyLessThan(next, curr);
        } else if constexpr (Mode == SortMode::StrictIncreasing) {
            return Traits::AnyLessEqual(next, curr);
        } else if constexpr (Mode == SortMode::Decreasing) {
            return Traits::AnyLessThan(curr, next);
        } else {
            return Traits::AnyLessEqual(curr, next);
        }
    }

    template <class Traits, SortMode Mode>
    static bool Compare(auto next, auto curr, size_t len) noexcept {
        if constexpr (Mode == SortMode::Increasing) {
            return Traits::AnyLessThan(next, curr, len);
        } else if constexpr (Mode == SortMode::StrictIncreasing) {
            return Traits::AnyLessEqual(next, curr, len);
        } else if constexpr (Mode == SortMode::Decreasing) {
            return Traits::AnyLessThan(curr, next, len);
        } else {
            return Traits::AnyLessEqual(curr, next, len);
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

    template <SortMode Mode, typename T>
    bool IsSorted(const T* data, size_t len) noexcept {
        const T* begin = data;
        const T* end = data + len;
        using ElemType = decltype(ToArithmeticType<T>());
        constexpr size_t ElemBytes = sizeof(T);

        constexpr int VecBits{
#if defined(__AVX512F__)
                512
#elif defined(__AVX__)
                256
#elif defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
                128
#endif
        };
        constexpr size_t VecElems = VecBits / 8 / ElemBytes;

        constexpr auto comp = ToCompareType<Mode>();

        if constexpr (VecBits > 0) {
            using Traits = VecTraits<VecBits, ElemType>;
            if constexpr (VecElems == 2) {
                while (len > 2) {
                    const auto curr = Traits::Load(begin);
                    const auto next = Traits::Load(begin + 1);
                    if (Compare<Traits, Mode>(next, curr)) {
                        return false;
                    }
                    begin += 2;
                    len -= 2;
                }
                return !(len > 1 && comp(*(begin + 1), *begin));
            } else {
                if (len > VecElems) {
                    const auto curr = Traits::Load(begin);
                    const auto next = Traits::Load(begin + 1);
                    if (Compare<Traits, Mode>(next, curr)) {
                        return false;
                    }
                    begin += (len - 1 + VecElems - 1) % VecElems + 1;
                    for (; begin != end - 1; begin += VecElems) {
                        const auto curr = Traits::Load(begin);
                        const auto next = Traits::Load(begin + 1);
                        if (Compare<Traits, Mode>(next, curr)) {
                            return false;
                        }
                    }
                    return true;
                }
                if (len > 1) {
                    const auto curr = Traits::Load(begin);
                    const auto next = Traits::Load(begin + 1);
                    if (Compare<Traits, Mode>(next, curr, len - 1)) {
                        return false;
                    }
                }
                return true;
            }
        } else {
            if (len > 1) {
                auto next = begin + 1;
                do {
                    if (comp(*next, *begin)) {
                        return false;
                    }
                    begin = next;
                    ++next;
                } while (next != end);
            }
            return true;
        }
    }
}  // namespace detail


template <detail::CanVectorize T>
bool IsIncreasing(const T* data, size_t len) noexcept {
    return detail::IsSorted<detail::SortMode::Increasing>(data, len);
}

template <detail::CanVectorize T>
bool IsStrictIncreasing(const T* data, size_t len) noexcept {
    return detail::IsSorted<detail::SortMode::StrictIncreasing>(data, len);
}

template <detail::CanVectorize T>
bool IsDecreasing(const T* data, size_t len) noexcept {
    return detail::IsSorted<detail::SortMode::Decreasing>(data, len);
}

template <detail::CanVectorize T>
bool IsStrictDecreasing(const T* data, size_t len) noexcept {
    return detail::IsSorted<detail::SortMode::StrictDecreasing>(data, len);
}
