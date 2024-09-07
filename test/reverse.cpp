#include <cassert>

#include <algorithm>
#include <format>
#include <iostream>
#include <vector>

#include "src/reverse.h"


int main() {
    std::vector<int16_t> arr;
    for (int i = 0; i < 10; ++i) {
        arr.push_back(i);
    }
    Reverse(arr.data(), arr.size());
    for (auto i: arr) {
        std::cout << i << " ";
    }
}
