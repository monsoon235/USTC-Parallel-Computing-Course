#include <iostream>
#include <iomanip>
#include <tuple>
#include <omp.h>
#include <mpi.h>


int count_prime_mpi(int n) {
    return 0;
}

int count_prime_omp(int n) {
    return 0;
}

std::tuple<double, double> calculate_pi_mpi(int thread_num, int max_iteration) {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = MPI_Wtime();
    double pi_local = 0;
    double pi = 0;
    double step = 1.0 / max_iteration;
    for (int i = rank; i < max_iteration; i += size) {
        double x = (i + 0.5) * step;
        pi_local += 4.0 / (1.0 + x * x);
    }
    MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    pi /= max_iteration;
    double end_time = MPI_Wtime();
    MPI_Finalize();
    return std::make_tuple(pi, end_time - start_time);
}

std::tuple<double, double> calculate_pi_omp(int thread_num, int max_iteration) {
    double pi = 0;
    double start_time = omp_get_wtime();
    double step = 1.0 / max_iteration; // 减少除法运算
    omp_set_num_threads(thread_num);
#pragma omp parallel for reduction(+ : pi)
    for (int i = 0; i < max_iteration; i++) {
        double x = (i + 0.5) * step;
        pi += 4.0 / (1 + x * x);
    }
    pi /= max_iteration;
    double end_time = omp_get_wtime();
    return std::make_tuple(pi, end_time - start_time);
}

int main() {
    auto thread_num_list = {1, 2, 4, 8};
    auto max_iteration_list = {1000, 10000, 50000, 100000};
    double pi, time;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(15);
    std::cout << "OpenMP 计算 pi:\n";
    std::cout << "线程数\t迭代次数\t用时(ms)\t\t\t\t结果\n";
    for (auto thread_num:thread_num_list) {
        for (auto max_iteration:max_iteration_list) {
            std::tie(pi, time) = calculate_pi_omp(thread_num, max_iteration);
            std::cout << thread_num << "\t\t" << max_iteration << '\t' << time * 1e3 << '\t' << pi << '\n';
        }
    }
    std::cout << std::endl;
    std::cout << "MPI 计算 pi:\n";
    std::cout << "线程数\t迭代次数\t用时(ms)\t\t\t\t结果\n";
    for (auto thread_num:thread_num_list) {
        for (auto max_iteration:max_iteration_list) {
            std::tie(pi, time) = calculate_pi_mpi(thread_num, max_iteration);
            std::cout << thread_num << "\t\t" << max_iteration << '\t' << time * 1e3 << '\t' << pi << '\n';
        }
    }
    return 0;
}
