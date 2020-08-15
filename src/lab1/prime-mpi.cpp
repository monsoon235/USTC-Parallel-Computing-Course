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

std::tuple<int, double> count_prime_mpi(int max) {

    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto is_prime = new bool[max + 1];
    std::fill_n(is_prime, max + 1, true);

    // 先找出 1~√max 的素数
    int stop = std::sqrt(max);
    mark_prime(is_prime, stop);

    // 数据分段
    int start_index = rank * (max + 1) / size;
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

    // 统计
    start_index = start_index / 2 * 2 + 1;// 只统计奇数
    int count = 0;
    int count_local = 0;
    for (int i = start_index; i < end_index; i += 2) {
        if (is_prime[i]) {
            count_local++;
        }
    }
    MPI_Reduce(&count_local, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (max >= 2) {
        count++;
    }

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
