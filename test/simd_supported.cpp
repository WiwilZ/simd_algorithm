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


    return 0;
}
