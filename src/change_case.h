/*
 * Created by WiwilZ on 2022/7/13.
 */
#pragma once


#include <cstring>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cstddef>
#include <cstdint>

#include <concepts>

#if defined(_MSC_VER) && !defined(__clang__)
#define __restrict__ __restrict
#endif


namespace detail {
    enum class CaseMode {
        Lower,
        Upper,
        Flip
    };

    template <size_t NumBits, bool IsAllAlpha, CaseMode Mode>
    struct VecTraits;


    struct Vec64TraitsBase {
        using Type = uint64_t;

        static constexpr uint64_t mask = 0x2020'2020'2020'2020;

        static Type Load(const void* data) noexcept {
            return *static_cast<const Type*>(data);
        }

        static void Store(void* data, Type value) noexcept {
            *static_cast<Type*>(data) = value;
        }
    };

    template <CaseMode Mode>
    struct VecTraits<64, true, Mode> : Vec64TraitsBase {
        static Type Transform(Type v) noexcept {
            if constexpr (Mode == CaseMode::Lower) {
                return v | mask;
            } else if constexpr (Mode == CaseMode::Upper) {
                return v & ~mask;
            } else {
                return v ^ mask;
            }
        }
    };

    template <CaseMode Mode>
    struct VecTraits<64, false, Mode> : Vec64TraitsBase {
        static constexpr uint64_t l_offset = Mode == CaseMode::Lower ? 0x3F3F'3F3F'3F3F'3F3F : 0x1F1F'1F1F'1F1F'1F1F;// 0x3F = -('A' - 1) + 127  0x1F = -('a' - 1) + 127
        static constexpr uint64_t r_offset = Mode == CaseMode::Lower ? 0x2525'2525'2525'2525 : 0x0505'0505'0505'0505;// 0x25 = -'Z' + 127        0x05 = -'z' + 127

        static Type Transform(Type v) noexcept {
            // c >= 'a' => c - ('a' - 1) + 127 > 127 => msb = 1
            // c <= 'z' => c - 'z' + 127 <= 127 => msb = 0
            Type t = v;
            if constexpr (Mode == CaseMode::Flip) {
                t |= mask;
            }
            return v ^ ((((t + l_offset) & ~(t + r_offset)) >> 2) & mask);
        }
    };

#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
    struct Vec128TraitsBase {
        using Type = __m128i;

        static inline const auto mask = _mm_set1_epi8(0x20);

        static Type Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm_storeu_si128(static_cast<__m128i*>(data), value);
        }
    };

    template <CaseMode Mode>
    struct VecTraits<128, true, Mode> : Vec128TraitsBase {
        static Type Transform(Type v) noexcept {
            if constexpr (Mode == CaseMode::Lower) {
                return _mm_or_si128(v, mask);
            } else if constexpr (Mode == CaseMode::Upper) {
                return _mm_andnot_si128(mask, v);
            } else {
                return _mm_xor_si128(v, mask);
            }
        }
    };

    template <CaseMode Mode>
    struct VecTraits<128, false, Mode> : Vec128TraitsBase {
        static inline const auto offset = _mm_set1_epi8((Mode == CaseMode::Lower ? 'A' : 'a') + 128);
        static inline const auto limit = _mm_set1_epi8(26 - 128);

        static Type Transform(Type v) noexcept {
            const auto dist = _mm_sub_epi8(Mode == CaseMode::Flip ? _mm_or_si128(v, mask) : v, offset);
            const auto is_alpha_mask = _mm_cmplt_epi8(dist, limit);
            const auto flip_mask = _mm_and_si128(is_alpha_mask, mask);
            return _mm_xor_si128(v, flip_mask);
        }
    };
#endif

#ifdef __AVX2__
    struct Vec256TraitsBase {
        using Type = __m256i;

        static inline const auto mask = _mm256_set1_epi8(0x20);

        static Type Load(const void* data) noexcept {
            return _mm256_lddqu_si256(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm256_storeu_si256(static_cast<__m256i*>(data), value);
        }
    };

    template <CaseMode Mode>
    struct VecTraits<256, true, Mode> : Vec256TraitsBase {
        static Type Transform(Type v) noexcept {
            if constexpr (Mode == CaseMode::Lower) {
                return _mm256_or_si256(v, mask);
            } else if constexpr (Mode == CaseMode::Upper) {
                return _mm256_andnot_si256(mask, v);
            } else {
                return _mm256_xor_si256(v, mask);
            }
        }
    };

    template <CaseMode Mode>
    struct VecTraits<256, false, Mode> : Vec256TraitsBase {
        static inline const auto offset = _mm256_set1_epi8((Mode == CaseMode::Lower ? 'A' : 'a') + 128);
        static inline const auto limit = _mm256_set1_epi8(26 - 128);

        static Type Transform(Type v) noexcept {
            const auto dist = _mm256_sub_epi8(Mode == CaseMode::Flip ? _mm256_or_si256(v, mask) : v, offset);
            const auto is_alpha_mask = _mm256_cmpgt_epi8(limit, dist);
            const auto flip_mask = _mm256_and_si256(is_alpha_mask, mask);
            return _mm256_xor_si256(v, flip_mask);
        }
    };
#endif

#ifdef __AVX512F__
    struct Vec512TraitsBase {
        using Type = __m512i;

        static inline const auto mask = _mm512_set1_epi8(0x20);

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }

        static void Store(void* data, Type value) noexcept {
            _mm512_storeu_si512(data, value);
        }
    };

    template <CaseMode Mode>
    struct VecTraits<512, true, Mode> : Vec512TraitsBase {
        static Type Transform(Type v) noexcept {
            if constexpr (Mode == CaseMode::Lower) {
                return _mm512_or_si512(v, mask);
            } else if constexpr (Mode == CaseMode::Upper) {
                return _mm512_andnot_si512(mask, v);
            } else {
                return _mm512_xor_si512(v, mask);
            }
        }
    };

#ifdef __AVX512BW__
    template <CaseMode Mode>
    struct VecTraits<512, false, Mode> : Vec512TraitsBase {
        static inline const auto offset = _mm512_set1_epi8((Mode == CaseMode::Lower ? 'A' : 'a') + 128);
        static inline const auto limit = _mm512_set1_epi8(26 - 128);

        static Type Transform(Type v) noexcept {
            const auto dist = _mm512_sub_epi8(Mode == CaseMode::Flip ? _mm512_or_si512(v, mask) : v, offset);
            const auto is_alpha_mask = _mm512_cmplt_epi8_mask(dist, limit);
            const auto flip_mask = _mm512_maskz_mov_epi8(is_alpha_mask, mask);
            return _mm512_xor_si512(v, flip_mask);
        }
    };
#endif
#endif


    template <class Traits>
    void Impl(char* const __restrict__ dst, const char* const __restrict__ src, size_t len) noexcept {
        using VecTy = typename Traits::Type;
        if (len >= sizeof(VecTy)) {
            auto* pd = reinterpret_cast<VecTy*>(dst);
            const auto* first = reinterpret_cast<const VecTy*>(src);
            const auto* const last = reinterpret_cast<const VecTy*>(src + len);
            for (; first + 1 <= last; ++first, ++pd) {
                Traits::Store(pd, Traits::Transform(Traits::Load(first)));
            }
            if (first != last) {
                Traits::Store(reinterpret_cast<VecTy*>(dst + len) - 1, Traits::Transform(Traits::Load(last - 1)));
            }
            return;
        }
        const auto v = Traits::Transform(Traits::Load(src));
#ifdef __AVX512BW__
        _mm512_mask_storeu_epi8(dst, (static_cast<__mmask64>(1) << len) - 1, v);
#else
        char tmp[sizeof(VecTy)];
        Traits::Store(tmp, v);
        std::memcpy(dst, tmp, len);
#endif
    }


    template <bool IsAllAlpha, CaseMode Mode>
    void Helper(char* const __restrict__ dst, const char* const __restrict__ src, size_t len) noexcept {
#if defined(__AVX512BW__)
        constexpr size_t NumBits = 512;
#elif defined(__AVX512F__)
        constexpr size_t NumBits = IsAllAlpha ? 512 : 256;
#elif defined(__AVX2__)
        constexpr size_t NumBits = 256;
#elif defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
        constexpr size_t NumBits = 128;
#else
        constexpr size_t NumBits = 64;
#endif
        return Impl<VecTraits<NumBits, IsAllAlpha, Mode>>(dst, src, len);
    }
}// namespace detail


template <bool IsAllAlpha = false>
void ToLower(char* __restrict__ dst, const char* __restrict__ src, size_t len) noexcept {
    detail::Helper<IsAllAlpha, detail::CaseMode::Lower>(dst, src, len);
}

template <bool IsAllAlpha = false>
void ToUpper(char* __restrict__ dst, const char* __restrict__ src, size_t len) noexcept {
    detail::Helper<IsAllAlpha, detail::CaseMode::Upper>(dst, src, len);
}

template <bool IsAllAlpha = false>
void ToFlip(char* __restrict__ dst, const char* __restrict__ src, size_t len) noexcept {
    detail::Helper<IsAllAlpha, detail::CaseMode::Flip>(dst, src, len);
}
