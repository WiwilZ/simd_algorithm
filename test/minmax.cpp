#include <format>
#include <iostream>
#include <vector>

#include "src/minmax.h"


int main() {
    std::vector<int64_t> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto [min, max] = MinMax(v.data(), v.size());
    std::cout << std::format("min: {}, max: {}\n", min, max);

    return 0;
}
