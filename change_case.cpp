/*
 * Created by WiwilZ on 2022/7/13.
 */

#include "src/change_case.h"
#include "benchmark/include/benchmark/benchmark.h"
#include <cstring>
#include <iostream>
#include <queue>


std::string_view str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! \"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                       "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! \"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                       "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! \"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                       "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! \"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                       "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ! \"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
std::string_view str_all_alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
inline constexpr size_t size = 1000;
char result[size];


static void BM_to_lower_not_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = 'A' <= str[i] && str[i] <= 'Z' ? str[i] ^ 0x20 : str[i];
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_lower_not_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_to_lower_not_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        to_lower(result, str.data(), str.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_lower_not_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


static void BM_to_upper_not_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = 'a' <= str[i] && str[i] <= 'z' ? str[i] ^ 0x20 : str[i];
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_upper_not_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_to_upper_not_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        to_upper(result, str.data(), str.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_upper_not_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


static void BM_flip_case_not_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = 'A' <= str[i] && str[i] <= 'Z' || 'a' <= str[i] && str[i] <= 'z' ? str[i] ^ 0x20 : str[i];
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_flip_case_not_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_flip_case_not_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        flip_case(result, str.data(), str.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_flip_case_not_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


static void BM_to_lower_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = str[i] | 0x20;
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_lower_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_to_lower_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        to_lower<true>(result, str_all_alpha.data(), str_all_alpha.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str_all_alpha);
    }
}

BENCHMARK(BM_to_lower_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


static void BM_to_upper_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = str[i] & ~0x20;
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_to_upper_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_to_upper_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        to_upper<true>(result, str_all_alpha.data(), str_all_alpha.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str_all_alpha);
    }
}

BENCHMARK(BM_to_upper_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


static void BM_flip_case_all_alpha_naive(benchmark::State& state) {
    for (auto _: state) {
        for (size_t i = 0; i < str.size(); i++) {
            result[i] = str[i] ^ 0x20;
        }
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str);
    }
}

BENCHMARK(BM_flip_case_all_alpha_naive)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);

static void BM_flip_case_all_alpha(benchmark::State& state) {
    for (auto _: state) {
        flip_case<true>(result, str_all_alpha.data(), str_all_alpha.size());
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(str_all_alpha);
    }
}

BENCHMARK(BM_flip_case_all_alpha)->Iterations(99901)->Repetitions(199)->DisplayAggregatesOnly(true);


//BENCHMARK_MAIN();


int main() {
    //	to_lower(result, str.data(), str.size());
    //	std::cout << result << std::endl;
    //
    //	to_upper(result, str.data(), str.size());
    //	std::cout << result << std::endl;
    //
    //	flip_case(result, str.data(), str.size());
    //	std::cout << result << std::endl;
    //
    //	memset(result, 0, size);
    //
    //	to_lower<true>(result, str_all_alpha.data(), str_all_alpha.size());
    //	std::cout << result << std::endl;
    //
    //	to_upper<true>(result, str_all_alpha.data(), str_all_alpha.size());
    //	std::cout << result << std::endl;
    //
    //	flip_case<true>(result, str_all_alpha.data(), str_all_alpha.size());
    //	std::cout << result << std::endl;


    return 0;
}