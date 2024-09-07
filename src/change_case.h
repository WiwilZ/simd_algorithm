#pragma once


#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <concepts>


namespace detail {
    enum class CaseMode {
        Lower,
        Upper,
        Flip
    };

    template <CaseMode Mode>
    constexpr char LowerBound = Mode == CaseMode::Lower ? 'A' : 'a';

    template <size_t VecBits, bool IsAllAlpha, CaseMode Mode>
    struct VecTraits;

    template <typename T, bool IsAllAlpha, CaseMode Mode>
    struct VeciTraitsBase {
        using Type = T;

        static constexpr Type mask = 0x2020'2020'2020'2020;
        static constexpr Type l_offset = Mode == CaseMode::Lower ? 0x3F3F'3F3F'3F3F'3F3F : 0x1F1F'1F1F'1F1F'1F1F;  // 0x3F = -('A' - 1) + 127  0x1F = -('a' - 1) + 127
        static constexpr Type r_offset = Mode == CaseMode::Lower ? 0x2525'2525'2525'2525 : 0x0505'0505'0505'0505;  // 0x25 = -'Z' + 127        0x05 = -'z' + 127

        static Type Load(const void* data) noexcept {
            return *static_cast<const Type*>(data);
        }

        static void Store(void* data, Type value) noexcept {
            *static_cast<Type*>(data) = value;
        }

        static constexpr Type Convert(Type v) noexcept {
            if constexpr (IsAllAlpha) {
                if constexpr (Mode == CaseMode::Lower) {
                    return v | mask;
                } else if constexpr (Mode == CaseMode::Upper) {
                    return v & ~mask;
                } else {
                    return v ^ mask;
                }
            } else {
                const auto t = Mode == CaseMode::Flip ? v | mask : v;
                if constexpr (sizeof(Type) == 1) {
                    return v ^ ((static_cast<uint8_t>(t - LowerBound<Mode>) < 26) << 5);
                } else {
                    // c >= 'a' => c - ('a' - 1) + 127 > 127 => msb = 1
                    // c <= 'z' => c - 'z' + 127 <= 127 => msb = 0
                    return v ^ ((((t + l_offset) & ~(t + r_offset)) >> 2) & mask);
                }
            }
        }
    };

    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<8, IsAllAlpha, Mode> : VeciTraitsBase<uint8_t, IsAllAlpha, Mode> {};

    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<16, IsAllAlpha, Mode> : VeciTraitsBase<uint16_t, IsAllAlpha, Mode> {};

    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<32, IsAllAlpha, Mode> : VeciTraitsBase<uint32_t, IsAllAlpha, Mode> {};

    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<64, IsAllAlpha, Mode> : VeciTraitsBase<uint64_t, IsAllAlpha, Mode> {};


#if defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<128, IsAllAlpha, Mode> {
        using Type = __m128i;

        static inline const auto mask = _mm_set1_epi8(0x20);
        static inline const auto offset = _mm_set1_epi8(LowerBound<Mode> + 128);
        static inline const auto limit = _mm_set1_epi8(26 - 128);

        static Type Load(const void* data) noexcept {
            return _mm_loadu_si128(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm_storeu_si128(static_cast<Type*>(data), value);
        }

        static Type Convert(Type v) noexcept {
            if constexpr (IsAllAlpha) {
                if constexpr (Mode == CaseMode::Lower) {
                    return _mm_or_si128(mask, v);
                } else if constexpr (Mode == CaseMode::Upper) {
                    return _mm_andnot_si128(mask, v);
                } else {
                    return _mm_xor_si128(mask, v);
                }
            } else {
                const auto dist = _mm_sub_epi8(Mode == CaseMode::Flip ? _mm_or_si128(v, mask) : v, offset);
                const auto alpha_mask = _mm_cmpgt_epi8(limit, dist);
                return _mm_xor_si128(v, _mm_and_si128(alpha_mask, mask));
            }
        }
    };
#endif

#ifdef __AVX__
    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<256, IsAllAlpha, Mode> {
        using Type = __m256i;

        static inline const auto mask = _mm256_set1_epi8(0x20);
        static inline const auto offset = _mm256_set1_epi8(LowerBound<Mode> + 128);
        static inline const auto limit = _mm256_set1_epi8(26 - 128);

        static Type Or(Type a, Type b) noexcept {
#ifdef __AVX2__
            return _mm256_or_si256(a, b);
#else
            return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
        }

        static Type And(Type a, Type b) noexcept {
#ifdef __AVX2__
            return _mm256_and_si256(a, b);
#else
            return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
        }

        static Type AndNot(Type a, Type b) noexcept {
#ifdef __AVX2__
            return _mm256_andnot_si256(a, b);
#else
            return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
        }

        static Type Xor(Type a, Type b) noexcept {
#ifdef __AVX2__
            return _mm256_xor_si256(a, b);
#else
            return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
        }

        static Type Load(const void* data) noexcept {
            return _mm256_loadu_si256(static_cast<const Type*>(data));
        }

        static void Store(void* data, Type value) noexcept {
            _mm256_storeu_si256(static_cast<Type*>(data), value);
        }

        static Type Convert(Type v) noexcept {
            if constexpr (IsAllAlpha) {
                if constexpr (Mode == CaseMode::Lower) {
                    return Or(mask, v);
                } else if constexpr (Mode == CaseMode::Upper) {
                    return AndNot(mask, v);
                } else {
                    return Xor(mask, v);
                }
            } else {
                const auto t = Mode == CaseMode::Flip ? Or(v, mask) : v;
#ifdef __AVX2__
                const auto dist = _mm256_sub_epi8(t, offset);
                const auto alpha_mask = _mm256_cmpgt_epi8(limit, dist);
#else
                const auto dist_low = _mm_sub_epi8(_mm256_castsi256_si128(t), VecTraits<128, IsAllAlpha, Mode>::offset);
                const auto dist_high = _mm_sub_epi8(_mm256_extractf128_si256(t, 1), VecTraits<128, IsAllAlpha, Mode>::offset);
                const auto alpha_mask_low = _mm_cmpgt_epi8(VecTraits<128, IsAllAlpha, Mode>::limit, dist_low);
                const auto alpha_mask_high = _mm_cmpgt_epi8(VecTraits<128, IsAllAlpha, Mode>::limit, dist_high);
                const auto alpha_mask = _mm256_insertf128_si256(_mm256_castsi128_si256(alpha_mask_low), alpha_mask_high, 1);
#endif
                return Xor(v, And(alpha_mask, mask));
            }
        }
    };
#endif

#ifdef __AVX512F__
    template <bool IsAllAlpha, CaseMode Mode>
    struct VecTraits<512, IsAllAlpha, Mode> {
        using Type = __m512i;

        static inline const auto mask = _mm512_set1_epi8(0x20);
        static inline const auto offset = _mm512_set1_epi8(LowerBound<Mode>);
        static inline const auto limit = _mm512_set1_epi8(26);

        static Type Load(const void* data) noexcept {
            return _mm512_loadu_si512(data);
        }

        static void Store(void* data, Type value) noexcept {
            _mm512_storeu_si512(data, value);
        }

        static Type Convert(Type v) noexcept {
            if constexpr (IsAllAlpha) {
                if constexpr (Mode == CaseMode::Lower) {
                    return _mm512_or_si512(mask, v);
                } else if constexpr (Mode == CaseMode::Upper) {
                    return _mm512_andnot_si512(mask, v);
                } else {
                    return _mm512_xor_si512(mask, v);
                }
            } else {
                const auto t = Mode == CaseMode::Flip ? _mm512_or_si512(v, mask) : v;
#ifdef __AVX512BW__
                const auto dist = _mm512_sub_epi8(t, offset);
                const auto alpha_mask = _mm512_cmplt_epu8_mask(dist, limit);
                return _mm512_mask_blend_epi8(alpha_mask, v, _mm512_xor_si512(v, mask));
#else
                const auto dist_low = _mm256_sub_epi8(_mm512_castsi512_si256(t), VecTraits<256, IsAllAlpha, Mode>::offset);
                const auto dist_high = _mm256_sub_epi8(_mm512_extracti64x4_epi64(t, 1)), VecTraits<256, IsAllAlpha, Mode>::offset);
                const auto alpha_mask_low = _mm256_cmpgt_epi8(VecTraits<256, IsAllAlpha, Mode>::limit, dist_low);
                const auto alpha_mask_high = _mm256_cmpgt_epi8(VecTraits<256, IsAllAlpha, Mode>::limit, dist_high);
                const auto alpha_mask = _mm512_inserti64x4(_mm512_castsi256_si512(alpha_mask_low), alpha_mask_high, 1);
                return _mm512_xor_si512(v, _mm512_and_si512(alpha_mask, mask));
#endif
            }
        }

#ifdef __AVX512BW__
        static Type GetAlphaMask(Type v) noexcept {
            const auto dist = _mm512_sub_epi8(Mode == CaseMode::Flip ? _mm512_or_si512(v, mask) : v, offset);
            return _mm512_cmplt_epu8_mask(dist, limit);
        }

        static void ImplNotAllAlpha(void* data) noexcept {
            const auto v = Load(data);
            _mm512_mask_storeu_epi8(data, GetAlphaMask(v), _mm512_xor_si512(v, mask));
        }

        static void ImplNotAllAlpha(void* data, size_t len) noexcept {
            const auto v = Load(data);
            _mm512_mask_storeu_epi8(data, GetAlphaMask(v) & (static_cast<__mmask64>(-1) >> (64 - len)), _mm512_xor_si512(v, mask));
        }
#endif
    };
#endif


    template <class Traits>
    void Impl1xVec(void* dst, const void* src) noexcept {
        Traits::Store(dst, Traits::Convert(Traits::Load(src)));
    }

    template <class Traits>
    void ImplGt1Lt2xVec(void* dst, void* dst_end, const void* src, const void* src_end) noexcept {
        using VecType = typename Traits::Type;
        const auto v = Traits::Convert(Traits::Load(static_cast<const VecType*>(src_end) - 1));
        Impl1xVec<Traits>(dst, src);
        Traits::Store(static_cast<VecType*>(dst_end) - 1, v);
    }


    template <bool IsAllAlpha, CaseMode Mode>
    void Impl(char* dst, const char* src, size_t len) noexcept {
        char* dst_end = dst + len;
        const char* src_end = src + len;

        constexpr int VecBits{
#if defined(__AVX512BW__)
                512
#elif defined(__AVX512F__)
                IsAllAlpha ? 512 : 256
#elif defined(__AVX2__)
                256
#elif defined(__AVX__)
                IsAllAlpha ? 256 : 128
#elif defined(__SSE2__) || defined(_MSC_VER) && !defined(__clang__) && ((defined(_M_AMD64) || defined(_M_X64)) && !defined(_M_ARM64EC) || defined(_M_IX86_FP) && _M_IX86_FP == 2)
                128
#else
                64
#endif
        };

        using Traits = VecTraits<VecBits, IsAllAlpha, Mode>;
        constexpr size_t VecElems = VecBits / 8;

        if (len >= VecElems) {
            using VecType = typename Traits::Type;
            auto* dst_first = reinterpret_cast<VecType*>(dst);
            auto* dst_last = reinterpret_cast<VecType*>(dst_end) - 1;
            const auto* src_first = reinterpret_cast<const VecType*>(src);
            const auto* src_last = reinterpret_cast<const VecType*>(src_end) - 1;
            Impl1xVec<Traits>(dst_last, src_last);
            for (; src_first < src_last; ++src_first, ++dst_first) {
                Impl1xVec<Traits>(dst_first, src_first);
            }
            return;
        }
#ifdef __AVX512BW__
        if (len > 0) {
            _mm512_mask_storeu_epi8(dst, static_cast<__mmask64>(-1) >> (64 - len), Traits::Convert(Traits::Load(src));
        }
#else
#ifdef __AVX512F__
        if (len >= 4) {
            using Traits1 = VecTraits<32, IsAllAlpha, Mode>;
            const auto tmp = Traits1::Load(src_end - 4);
            _mm512_mask_storeu_epi32(dst, static_cast<__mmask16>(-1) >> (16 - (len >> 2)), Traits::Convert(Traits::Load(src));
            Traits1::Store(dst_end - 4, Traits1::Convert(tmp));
            return;
        }
#else
        static_assert(VecElems <= 32);

        if constexpr (VecElems == 32) {
            if (len >= 16) {
                using Traits1 = VecTraits<128, IsAllAlpha, Mode>;
                if (len == 16) {
                    Impl1xVec<Traits1>(dst, src);
                } else {
                    ImplGt1Lt2xVec<Traits1>(dst, dst_end, src, src_end);
                }
                return;
            }
        }
        if constexpr (VecElems >= 16) {
            if (len >= 8) {
                using Traits1 = VecTraits<64, IsAllAlpha, Mode>;
                if (len == 8) {
                    Impl1xVec<Traits1>(dst, src);
                } else {
                    ImplGt1Lt2xVec<Traits1>(dst, dst_end, src, src_end);
                }
                return;
            }
        }
        if (len >= 4) {
            using Traits1 = VecTraits<32, IsAllAlpha, Mode>;
            if (len == 4) {
                Impl1xVec<Traits1>(dst, src);
            } else {
                ImplGt1Lt2xVec<Traits1>(dst, dst_end, src, src_end);
            }
            return;
        }
#endif
        if (len >= 2) {
            using Traits1 = VecTraits<16, IsAllAlpha, Mode>;
            if (len == 2) {
                Impl1xVec<Traits1>(dst, src);
            } else {
                ImplGt1Lt2xVec<Traits1>(dst, dst_end, src, src_end);
            }
            return;
        }
        if (len > 0) {
            Impl1xVec<VecTraits<8, IsAllAlpha, Mode>>(dst, src);
        }
#endif
    }

    template <bool IsAllAlpha, CaseMode Mode>
    void Impl(char* data, size_t len) noexcept {
        Impl<IsAllAlpha, Mode>(data, data, len);
    }

#ifdef __AVX512BW__
    template <bool IsAllAlpha, CaseMode Mode>
        requires(!IsAllAlpha)
    void Impl(char* data, size_t len) noexcept {
        using Traits = VecTraits<512, false, Mode>;
        using VecType = typename Traits::Type;
        if (len >= 64) {
            auto* first = reinterpret_cast<VecType*>(data);
            auto* last = reinterpret_cast<VecType*>(data + len) - 1;
            Traits::ImplNotAllAlpha(last);
            for (; first < last; ++first) {
                Traits::ImplNotAllAlpha(first);
            }
            return;
        }
        if (len > 0) {
            Traits::ImplNotAllAlpha(data, len);
        }
    }
#endif
}  // namespace detail


template <bool IsAllAlpha = false>
void ToLower(char* dst, const char* src, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Lower>(dst, src, len);
}

template <bool IsAllAlpha = false>
void ToUpper(char* dst, const char* src, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Upper>(dst, src, len);
}

template <bool IsAllAlpha = false>
void ToFlip(char* dst, const char* src, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Flip>(dst, src, len);
}

// inplace
template <bool IsAllAlpha = false>
void ToLower(char* data, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Lower>(data, len);
}

template <bool IsAllAlpha = false>
void ToUpper(char* data, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Upper>(data, len);
}

template <bool IsAllAlpha = false>
void ToFlip(char* data, size_t len) noexcept {
    detail::Impl<IsAllAlpha, detail::CaseMode::Flip>(data, len);
}
