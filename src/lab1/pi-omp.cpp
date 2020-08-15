#include <iostream>
#include <iomanip>
#include <tuple>
#include <omp.h>


std::tuple<double, double> calculate_pi_omp(int max_iteration, int thread_num) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(thread_num);

    double pi = 0;
    double step = 1.0 / max_iteration; // 减少除法运算

#pragma omp parallel for reduction(+ : pi)
    for (int i = 0; i < max_iteration; i++) {
        double x = (i + 0.5) * step;
        pi += 4.0 / (1 + x * x);
    }

    pi /= max_iteration;

    double end_time = omp_get_wtime();
    return std::make_tuple(pi, end_time - start_time);
}

int main(int argc, char *argv[]) {

    auto thread_num = std::stoi(argv[1]);
    auto max_iteration = std::stoi(argv[2]);

    std::cout << std::setiosflags(std::ios::fixed);
    std::cout << "OpenMP 计算 pi:\n";
    std::cout << "线程数\t迭代次数\t用时(ms)\t结果\n";
    double pi, time;
    std::tie(pi, time) = calculate_pi_omp(max_iteration, thread_num);
    std::cout << thread_num << "\t" << max_iteration << '\t'
              << std::setprecision(4) << time * 1e3 << '\t'
              << std::setprecision(15) << pi << '\n';
    return 0;
}
