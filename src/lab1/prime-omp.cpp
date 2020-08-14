#include <iostream>
#include <iomanip>
#include <tuple>
#include <cmath>
#include <omp.h>


std::tuple<int, double> count_prime_omp(int max, int thread_num) {

    if (max < 1) {
        return std::make_tuple(0, 0);
    }

    double start_time = omp_get_wtime();

    omp_set_num_threads(thread_num);

    auto is_prime = new bool[max + 1];
    std::fill_n(is_prime, max + 1, true);
    is_prime[0] = false;
    is_prime[1] = false;

    // 筛法
    // cache 不友好
    int stop = std::sqrt(max);
#pragma omp parallel for
    for (int i = 2; i <= stop; i++) {
        if (is_prime[i]) { // 由于 bool 只有 1 字节，它的读取天然保障是原子的，不需要再加锁
            for (int j = i * 2; j <= max; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // 只统计奇数
    int count = 0;
#pragma omp parallel for reduction(+:count)
    for (int i = 0; i < thread_num; i++) {
        int start_index = i * (max + 1) / thread_num;
        start_index = start_index / 2 * 2 + 1;
        int end_index = (i + 1) * (max + 1) / thread_num;
        if (i == thread_num - 1) {
            end_index = max + 1;
        }
        for (int j = start_index; j < end_index; j += 2) {
            if (is_prime[j]) {
                count++;
            }
        }
    }
    if (max >= 2) {
        count++;
    }

    delete[] is_prime;

    double end_time = omp_get_wtime();
    return std::make_tuple(count, end_time - start_time);
}

int main() {
    auto thread_num_list = {1, 2, 4, 8};
    auto max_list = {1000, 10000, 100000, 500000};

    std::cout << std::setiosflags(std::ios::fixed);
    std::cout << "OpenMP 统计素数个数:\n";
    std::cout << "线程数\t最大数\t用时(ms)\t结果\n";
    for (auto thread_num:thread_num_list) {
        for (auto max:max_list) {
            int count;
            double time;
            std::tie(count, time) = count_prime_omp(max, thread_num);
            std::cout << thread_num << "\t" << max << '\t'
                      << std::setprecision(4) << time * 1e3 << '\t'
                      << count << '\n';
        }
    }
    return 0;
}
