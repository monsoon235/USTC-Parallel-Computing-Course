#include <iostream>
#include <iomanip>
#include <tuple>
#include <cmath>
#include <string>
#include <mpi.h>


// 串行算法，小规模
void mark_prime(bool is_prime[], int max) {
    std::fill_n(is_prime, max + 1, true);
    is_prime[0] = false;
    if (max > 1) {
        is_prime[1] = false;
    }
    double stop = std::sqrt(max);
    for (int i = 2; i <= stop; i++) {
        if (is_prime[i]) {
            for (int j = i * 2; j < max; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

std::tuple<int, double> count_prime_mpi(int max) {

    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int offset = rank * (max + 1) / size;
    int length = (rank + 1) * (max + 1) / size - offset;

    // 先找出 1~√max 的素数
    int max_sqrt = std::sqrt(max);
    auto is_prime_before_sqrt = new bool[max_sqrt + 1];
    std::fill_n(is_prime_before_sqrt, max_sqrt + 1, true);
    mark_prime(is_prime_before_sqrt, max_sqrt);

    auto is_prime = new bool[length];  // 每个线程负责 [offset, offset+length) 的部分
    std::fill_n(is_prime, length, true);
    // 去除 0 1
    if (offset <= 0 && offset + length > 0) {
        is_prime[0 - offset] = false;
    }
    if (offset <= 1 && offset + length > 1) {
        is_prime[1 - offset] = false;
    }

    // 筛法
    for (int i = 2; i <= max_sqrt; i++) {
        if (is_prime_before_sqrt[i]) {
            int start = (offset % i == 0) ? 0 : (i - offset % i);
            int real_start = std::max(start, 2 * i - offset);
            for (int j = real_start; j < length; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // 统计，只统计奇数
    int count_local = 0;
    int start = 1 - offset % 2;
    for (int i = start; i < length; i += 2) {
        if (is_prime[i]) {
            count_local++;
        }
    }
    int count = 0;
    MPI_Reduce(&count_local, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (max >= 2) {
        count++;
    }

    delete[] is_prime_before_sqrt;
    delete[] is_prime;

    double end_time = MPI_Wtime();
    return std::make_tuple(count, end_time - start_time);
}


int main(int argc, char *argv[]) {

    auto max = std::stoi(argv[1]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "MPI 统计素数个数:\n";
        std::cout << "线程数\t最大数\t用时(ms)\t结果\n";
    }
    int count;
    double time;
    std::tie(count, time) = count_prime_mpi(max);
    if (rank == 0) {
        std::cout << size << "\t"
                  << max << '\t'
                  << std::setprecision(4) << time * 1e3 << '\t'
                  << count << '\n';
    }

    MPI_Finalize();
    return 0;
}
