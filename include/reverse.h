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
    template <size_t VecBits, size_t ElemBytes>
    struct VecTraits;

    template <typename T>
    struct VeciTraitsBase {
        using Type = T;

        static Type Load(const void* data) noexcept {
            return *static_cast<const Type*>(data);
        }

        static void Store(void* data, Type value) noexcept {
            *static_cast<Type*>(data) = value;
        }
    };

    template <>
    struct VecTraits<16, 1> : VeciTraitsBase<uint16_t> {
        static Type Reverse(Type v) noexcept {
            return (v << 8) | (v >> 8);
        }
    };

    template <>
    struct VecTraits<32, 1> : VeciTraitsBase<uint32_t> {
        static Type Reverse(Type v) noexcept {
            v = (v << 16) | (v >> 16);
            v = ((v & 0x00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF);
            return v;
        }
    };

    template <>
    struct VecTraits<32, 2> : VeciTraitsBase<uint32_t> {
        static Type Reverse(Type v) noexcept {
            return (v << 16) | (v >> 16);
        }
    };

    template <>
    struct VecTraits<64, 1> : VeciTraitsBase<uint64_t> {
        static Type Reverse(Type v) noexcept {
            v = (v << 32) | (v >> 32);
            v = ((v & 0x0000FFFF0000FFFF) << 16) | ((v >> 16) & 0x0000FFFF0000FFFF);
            v = ((v & 0x00FF00FF00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF00FF00FF);
            return v;
        }
    };

    template <>
    struct VecTraits<64, 2> : VeciTraitsBase<uint64_t> {
        static Type Reverse(Type v) noexcept {
            v = (v << 32) | (v >> 32);
            v = ((v & 0x0000FFFF0000FFFF) << 16) | ((v >> 16) & 0x0000FFFF0000FFFF);
            return v;
        }
    };

    template <>
    struct VecTraits<64, 4> : VeciTraitsBase<uint64_t> {
        static Type Reverse(Type v) noexcept {
            return (v << 32) | (v >> 32);
        }
    };

#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
    struct Vec128TraitsBase {
        using Type = __m128i;

        static Type Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm_storeu_si128(static_cast<Type*>(data), value);
        }
    };

    template <>
    struct VecTraits<128, 1> : Vec128TraitsBase {
        static inline const auto mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        static inline const auto zero = _mm_setzero_si128();

        static Type Reverse(Type v) noexcept {
#ifdef __SSSE3__
            return _mm_shuffle_epi8(v, mask);
#else
            auto t = _mm_unpacklo_epi8(v, zero);
            t = _mm_shuffle_epi32(t, _MM_SHUFFLE(1, 0, 3, 2));
            t = _mm_shufflelo_epi16(t, _MM_SHUFFLE(0, 1, 2, 3));
            t = _mm_shufflehi_epi16(t, _MM_SHUFFLE(0, 1, 2, 3));

            v = _mm_unpackhi_epi8(v, zero);
            v = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
            v = _mm_shufflelo_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
            v = _mm_shufflehi_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));

            return _mm_packus_epi16(v, t);
#endif
        }
    };

    template <>
    struct VecTraits<128, 2> : Vec128TraitsBase {
        static inline const auto mask = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        static Type Reverse(Type v) noexcept {
#ifdef __SSSE3__
            return _mm_shuffle_epi8(v, mask);
#else
            v = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
            v = _mm_shufflelo_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
            v = _mm_shufflehi_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
            return v;
#endif
        }
    };

    template <>
    struct VecTraits<128, 4> : Vec128TraitsBase {
        static Type Reverse(Type v) noexcept {
            return _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3));
        }
    };

    template <>
    struct VecTraits<128, 8> : Vec128TraitsBase {
        static Type Reverse(Type v) noexcept {
            return _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        }
    };
#endif

#ifdef __AVX__
    struct Vec256TraitsBase {
        using Type = __m256i;

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_si256(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm256_storeu_si256(static_cast<Type*>(data), value);
        }
    };

    template <size_t ElemBytes>
        requires(ElemBytes == 1 || ElemBytes == 2)
    struct VecTraits<256, ElemBytes> : Vec256TraitsBase {
        // clang-format off
        static inline const auto mask = ElemBytes == 1 ?
                _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) :
                _mm256_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        // clang-format on

        static Type Reverse(Type v) noexcept {
#ifdef __AVX2__
            return _mm256_permute4x64_epi64(_mm256_shuffle_epi8(v, mask), _MM_SHUFFLE(1, 0, 3, 2));
#else
            const auto low = VecTraits<128, ElemBytes>::Reverse(_mm256_castsi256_si128(v));
            const auto high = VecTraits<128, ElemBytes>::Reverse(_mm256_extractf128_si256(v, 1));
            return _mm256_insertf128_si256(_mm256_castsi128_si256(high), low, 1);
#endif
        }
    };

    template <>
    struct VecTraits<256, 4> : Vec256TraitsBase {
        static inline const auto index = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        static Type Reverse(Type v) noexcept {
#ifdef __AVX2__
            return _mm256_permutevar8x32_epi32(v, index);
#else
            const auto t = _mm256_castsi256_ps(_mm256_permute2f128_si256(v, v, 1));
            return _mm256_castps_si256(_mm256_shuffle_ps(t, t, _MM_SHUFFLE(0, 1, 2, 3)));
#endif
        }
    };

    template <>
    struct VecTraits<256, 8> : Vec256TraitsBase {
        static Type Reverse(Type v) noexcept {
#ifdef __AVX2__
            return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3));
#else
            const auto t = _mm256_castsi256_pd(_mm256_permute2f128_si256(v, v, 1));
            return _mm256_castpd_si256(_mm256_shuffle_pd(t, t, 5));
#endif
        }
    };

    template <>
    struct VecTraits<256, 16> : Vec256TraitsBase {
        static Type Reverse(Type v) noexcept {
            return _mm256_permute2f128_si256(v, v, 1);
        }
    };
#endif

#ifdef __AVX512F__
    struct Vec512TraitsBase {
        using Type = __m512i;

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }

        static void Store(void* data, Type value) noexcept {
            _mm512_storeu_si512(data, value);
        }
    };

#ifdef __AVX512BW__
    template <>
    struct VecTraits<512, 1> : Vec512TraitsBase {
        static inline const auto index = _mm512_set_epi8(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        static inline const auto mask = _mm512_set_epi8(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        static Type Reverse(Type v) noexcept {
#ifdef __AVX512VBMI__
            return _mm512_permutexvar_epi8(index, v);
#else
            v = _mm512_shuffle_epi8(v, mask);
            return _mm512_shuffle_i64x2(v, v, _MM_SHUFFLE(0, 1, 2, 3));
#endif
        }
    };

    template <>
    struct VecTraits<512, 2> : Vec512TraitsBase {
        static inline const auto index = _mm512_set_epi16(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

        static Type Reverse(Type v) noexcept {
            return _mm512_permutexvar_epi16(index, v);
        }
    };
#else
    template <size_t ElemBytes>
        requires(ElemBytes == 1 || ElemBytes == 2)
    struct VecTraits<512, ElemBytes> : Vec512TraitsBase {
        // clang-format off
        static inline const auto mask = ElemBytes == 1 ?
                _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) :
                _mm256_set_epi8(17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30,
                                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        // clang-format on

        static Type Reverse(Type v) noexcept {
            const auto low = _mm256_shuffle_epi8(_mm512_castsi512_si256(v), mask);
            const auto high = _mm256_shuffle_epi8(_mm512_extracti64x4_epi64(v, 1), mask);
            v = _mm512_inserti64x4(_mm512_castsi256_si512(high), low, 1);
            return _mm512_permutex_epi64(v, _MM_SHUFFLE(1, 0, 3, 2));
        }
    };
#endif

    template <>
    struct VecTraits<512, 4> : Vec512TraitsBase {
        static inline const auto index = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        static Type Reverse(Type v) noexcept {
            return _mm512_permutexvar_epi32(index, v);
        }
    };

    template <size_t ElemBytes>
        requires(ElemBytes == 8 || ElemBytes == 16 || ElemBytes == 32)
    struct VecTraits<512, ElemBytes> : Vec512TraitsBase {
        // clang-format off
        static inline const auto index = ElemBytes == 8 ? _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7) :
                                        ElemBytes == 16 ? _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6) :
                                                          _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        // clang-format on

        static Type Reverse(Type v) noexcept {
            return _mm512_permutexvar_epi64(index, v);
        }
    };
#endif


    constexpr int VecBits{
#if defined(__AVX512F__)
            512
#elif defined(__AVX__)
            256
#elif defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
            128
#else
            64
#endif
    };

    template <typename T, typename Elem = std::remove_reference_t<T>, size_t ElemBytes = sizeof(Elem)>
    concept CanVectorize = !std::is_volatile_v<Elem> && !requires(Elem& a, Elem& b) { { swap(a, b) }; }                                                                            //
                           && std::is_trivially_destructible_v<Elem> && std::is_trivially_move_constructible_v<Elem> && std::is_trivially_move_assignable_v<Elem>  //
                           && (ElemBytes & (ElemBytes - 1)) == 0 && ElemBytes <= VecBits / 8 / 2;


    template <class Traits>
    void ImplGt1Le2xVec(void* begin, void* end) noexcept {
        using VecType = typename Traits::Type;
        auto* last = static_cast<VecType*>(end) - 1;
        const auto left = Traits::Load(begin);
        Traits::Store(begin, Traits::Reverse(Traits::Load(last)));
        Traits::Store(last, Traits::Reverse(left));
    }

    template <class Traits>
    void ImplGe1Le2xVec(void* begin, void* end) noexcept {
        using VecType = typename Traits::Type;
        auto* first = static_cast<VecType*>(begin);
        auto* last = static_cast<VecType*>(end) - 1;
        const auto left = Traits::Load(first);
        if (first != last) {
            Traits::Store(first, Traits::Reverse(Traits::Load(last)));
        }
        Traits::Store(last, Traits::Reverse(left));
    }

    template <typename T>
    void Impl(T* data, size_t len) noexcept {
        T* end = data + len;
        constexpr size_t ElemBytes = sizeof(T);
        constexpr size_t VecElems = VecBits / 8 / ElemBytes;

        using Traits = VecTraits<VecBits, ElemBytes>;
        using VecType = typename Traits::Type;

        if constexpr (VecElems <= 4) {
            auto* first = reinterpret_cast<VecType*>(data);
            auto* last = reinterpret_cast<VecType*>(end);
            auto size = static_cast<intptr_t>(len);
            while (size >= 4) {
                ImplGt1Le2xVec<Traits>(first, last);
                ++first;
                --last;
                size -= 2 * VecElems;
            }
            if (size >= 2) {
                std::swap(*reinterpret_cast<T*>(first), *(reinterpret_cast<T*>(last) - 1));
            }
        } else {
            if (len >= VecElems) {
                const auto left = Traits::Load(data);
                if (len != VecElems) {
                    const auto right = Traits::Load(end - VecElems);
                    const size_t offset = (len / 2 + VecElems - 1) % VecElems + 1;
                    auto* first = reinterpret_cast<VecType*>(data + offset);
                    auto* last = reinterpret_cast<VecType*>(end - offset);
                    auto* sentinel = reinterpret_cast<VecType*>(data + len / 2);
                    for (; first != sentinel; ++first, --last) {
                        ImplGt1Le2xVec<Traits>(first, last);
                    }
                    Traits::Store(data, Traits::Reverse(right));
                }
                Traits::Store(end - VecElems, Traits::Reverse(left));
                return;
            }
#ifdef __AVX512F__
            if (len >= 4) {
                const auto v = Traits::Reverse(Traits::Load(data));
                const size_t shift = VecElems - len;
                if constexpr (VecElems == 8) {
                    // ElemBytes = 8
                    _mm512_mask_storeu_epi64(data - shift, static_cast<__mmask8>(-1) << shift, v);
                } else if constexpr (VecElems == 16) {
                    // ElemBytes = 4
                    _mm512_mask_storeu_epi32(data - shift, static_cast<__mmask16>(-1) << shift, v);
                } else if constexpr (VecElems == 32) {
                    // ElemBytes = 2
#ifdef __AVX512BW__
                    _mm512_mask_storeu_epi16(data - shift, static_cast<__mmask32>(-1) << shift, v);
#else
                    const auto tmp = *(end - 1);
                    _mm512_mask_storeu_epi32(data - shift, static_cast<__mmask16>(-1) << (16 - (len >> 1)), v);
                    *data = tmp;
#endif
                } else {
                    static_assert(VecElems == 64);
                    // ElemBytes = 1
#ifdef __AVX512BW__
                    _mm512_mask_storeu_epi8(data - shift, static_cast<__mmask64>(-1) << shift, v);
#else
                    using Traits1 = VecTraits<4, 1>;
                    const auto tmp = Traits1::Load(end - 4);
                    _mm512_mask_storeu_epi32(data - shift, static_cast<__mmask16>(-1) << (16 - (len >> 2)), v);
                    Traits1::Store(data, Traits1::Reverse(tmp));
#endif
                }
                return;
            }
#else
            static_assert(8 <= VecElems && VecElems <= 32);

            if constexpr (VecElems == 32) {
                if (len >= 16) {
                    ImplGe1Le2xVec<VecTraits<128 * ElemBytes, ElemBytes>>(data, end);
                    return;
                }
            }
            if constexpr (VecElems >= 16) {
                if (len >= 8) {
                    ImplGe1Le2xVec<VecTraits<64 * ElemBytes, ElemBytes>>(data, end);
                    return;
                }
            }
            if (len >= 4) {
                ImplGe1Le2xVec<VecTraits<32 * ElemBytes, ElemBytes>>(data, end);
                return;
            }
#endif
            if (len >= 2) {
                std::swap(*data, *(end - 1));
            }
        }
    }
}  // namespace detail


template <detail::CanVectorize T>
void Reverse(T* data, size_t len) noexcept {
    detail::Impl(data, len);
}
