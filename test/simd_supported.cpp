#include <iostream>

#include "src/simd_supported.h"


#define PRINT_SIMD_SUPPORTED(x) std::cout << #x << ": " << x << std::endl

int main() {
    PRINT_SIMD_SUPPORTED(simd_supported::ix86_fp);
    PRINT_SIMD_SUPPORTED(simd_supported::mmx);
    PRINT_SIMD_SUPPORTED(simd_supported::sse);
    PRINT_SIMD_SUPPORTED(simd_supported::sse2);
    PRINT_SIMD_SUPPORTED(simd_supported::sse3);
    PRINT_SIMD_SUPPORTED(simd_supported::sse4_1);
    PRINT_SIMD_SUPPORTED(simd_supported::sse4_2);
    PRINT_SIMD_SUPPORTED(simd_supported::avx);
    PRINT_SIMD_SUPPORTED(simd_supported::avx2);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512f);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512pf);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512er);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512cd);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vl);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512bw);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512dq);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512ifma);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vbmi);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vbmi2);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512bf16);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512fp16);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512bitalg);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vpopcntdq);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vp2intersect);
    PRINT_SIMD_SUPPORTED(simd_supported::avx5124fmaps);
    PRINT_SIMD_SUPPORTED(simd_supported::avx512vnni);
    PRINT_SIMD_SUPPORTED(simd_supported::avx5124vnniw);

    return 0;
}
