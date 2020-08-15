#include <iostream>
#include <iomanip>
#include <tuple>
#include <cmath>
#include <omp.h>


// 串行算法，小规模
void mark_prime(bool is_prime[], int max) {
    std::fill_n(is_prime, max + 1, true);
    is_prime[0] = false;
    is_prime[1] = false;
    double stop = std::sqrt(max);
    for (int i = 2; i <= stop; i++) {
        if (is_prime[i]) {
            for (int j = i * 2; j < max; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

std::tuple<int, double> count_prime_omp(int max, int thread_num) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(thread_num);

    auto is_prime = new bool[max + 1];
    std::fill_n(is_prime, max + 1, true);

    // 先找出 1~√max 的素数
    int stop = std::sqrt(max);
    mark_prime(is_prime, stop);

    // 筛法
#pragma omp parallel
    {
        int rank = omp_get_thread_num(); // 线程 id
        int size = omp_get_num_threads(); // 线程数
        int start_index = std::max(rank * (max + 1) / size, stop + 1);
        int end_index = (rank + 1) * (max + 1) / size;
        for (int i = 2; i <= stop; i++) {
            if (is_prime[i]) {
                // start_index 起第一个 i 的倍数
                int start = (start_index % i == 0) ? start_index : start_index + (i - start_index % i);
                for (int j = std::max(start, 2 * i); j < end_index; j += i) {
                    is_prime[j] = false;
                }
            }
        }
    }

    // 统计
    int count = max >= 2 ? 1 : 0;
#pragma omp parallel reduction(+:count)
    {
        int rank = omp_get_thread_num(); // 线程 id
        int size = omp_get_num_threads(); // 线程数
        int start_index = rank * (max + 1) / size;
        start_index = start_index / 2 * 2 + 1; // 只统计奇数
        int end_index = (rank + 1) * (max + 1) / size;
        for (int j = start_index; j < end_index; j += 2) {
            if (is_prime[j]) {
                count++;
            }
        }
    }

    delete[] is_prime;

    double end_time = omp_get_wtime();
    return std::make_tuple(count, end_time - start_time);
}

int main(int argc, char *argv[]) {

    auto thread_num = std::stoi(argv[1]);
    auto max = std::stoi(argv[2]);

    std::cout << std::setiosflags(std::ios::fixed);
    std::cout << "OpenMP 统计素数个数:\n";
    std::cout << "线程数\t最大数\t用时(ms)\t结果\n";
    int count;
    double time;
    std::tie(count, time) = count_prime_omp(max, thread_num);
    std::cout << thread_num << "\t" << max << '\t'
              << std::setprecision(4) << time * 1e3 << '\t'
              << count << '\n';
    return 0;
}
