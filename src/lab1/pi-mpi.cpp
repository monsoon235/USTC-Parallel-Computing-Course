#include <iostream>
#include <iomanip>
#include <tuple>
#include <string>
#include <mpi.h>


std::tuple<double, double> calculate_pi_mpi(int max_iter) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double pi_local = 0;
    double pi = 0;
    double step = 1.0 / max_iter;

    for (int i = rank; i < max_iter; i += size) {
        double x = (i + 0.5) * step;
        pi_local += 4.0 / (1.0 + x * x);
    }

    MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    pi /= max_iter;

    double end_time = MPI_Wtime();
    return std::make_tuple(pi, end_time - start_time);
}

int main(int argc, char *argv[]) {

    auto max_iter = std::stoi(argv[1]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "MPI 计算 pi:\n";
        std::cout << "线程数\t迭代次数\t用时(ms)\t结果\n";
    }
    double pi, time;
    std::tie(pi, time) = calculate_pi_mpi(max_iter);
    if (rank == 0) {
        std::cout << size << "\t"
                  << max_iter << '\t'
                  << std::setprecision(4) << time * 1e3 << '\t'
                  << std::setprecision(15) << pi << '\n';
    }

    MPI_Finalize();
    return 0;
}
