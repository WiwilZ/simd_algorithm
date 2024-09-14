#include "include/is_sorted.h"
#include <format>
#include <iostream>
#include <limits>
#include <vector>


int main() {
    std::vector<float> v = {1, 2, 3, std::numeric_limits<float>::quiet_NaN(), 4, 4, 5, 6, 7, 8, 9, 10};
    std::cout << std::boolalpha << IsIncreasing(v.data(), v.size()) << std::endl
              << IsStrictIncreasing(v.data(), v.size()) << std::endl;
    return 0;
}
