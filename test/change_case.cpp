#include <cassert>
#include <cstring>

#include <format>
#include <iostream>
#include <string>
#include <string_view>

#include "src/change_case.h"
#include <random>


void shuffle(auto& arr) {
    std::random_device rd;
    for (size_t i = arr.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(rd);
        std::swap(arr[i], arr[j]);
    }
}


int main() {
    std::string str;
    for (int i = 0; i < 128; i++) {
        str.push_back(i);
    }
    shuffle(str);
    for (int i = 0; i < 128; i++) {
        str.push_back(i);
    }
    for (int i = 127; i >= 0; --i) {
        str.push_back(i);
    }

    constexpr std::string_view alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string str_all_alpha;
    str_all_alpha.assign(alphabets.begin(), alphabets.end());
    shuffle(str_all_alpha);
    str_all_alpha.insert(str_all_alpha.end(), alphabets.begin(), alphabets.end());
    str_all_alpha.insert(str_all_alpha.end(), alphabets.rbegin(), alphabets.rend());


    constexpr size_t size = 4096;
    char result[size];

    std::memset(result, 0, size);
    ToLower<true>(result, str_all_alpha.data(), str_all_alpha.size());
    for (size_t i = 0; i < str_all_alpha.size(); i++) {
        assert(std::tolower(str_all_alpha[i]) == result[i]);
    }
    std::memset(result, 0, size);
    ToUpper<true>(result, str_all_alpha.data(), str_all_alpha.size());
    for (size_t i = 0; i < str_all_alpha.size(); i++) {
        assert(std::toupper(str_all_alpha[i]) == result[i]);
    }
    std::memset(result, 0, size);
    ToFlip<true>(result, str_all_alpha.data(), str_all_alpha.size());
    for (size_t i = 0; i < str_all_alpha.size(); i++) {
        char c = str_all_alpha[i];
        if ('a' <= c && c <= 'z') {
            c = std::toupper(c);
        } else if ('A' <= c && c <= 'Z') {
            c = std::tolower(c);
        }
        assert(c == result[i]);
    }

    std::memset(result, 0, size);
    ToLower(result, str.data(), str.size());
    for (size_t i = 0; i < str.size(); i++) {
        assert(std::tolower(str[i]) == result[i]);
    }
    std::memset(result, 0, size);
    ToUpper(result, str.data(), str.size());
    for (size_t i = 0; i < str.size(); i++) {
        assert(std::toupper(str[i]) == result[i]);
    }
    std::memset(result, 0, size);
    ToFlip(result, str.data(), str.size());
    for (size_t i = 0; i < str.size(); i++) {
        char c = str[i];
        if ('a' <= c && c <= 'z') {
            c = std::toupper(c);
        } else if ('A' <= c && c <= 'Z') {
            c = std::tolower(c);
        }
        assert(c == result[i]);
    }


    return 0;
}
