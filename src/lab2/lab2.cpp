#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <random>
#include <mpi.h>


struct Car {
    int velocity;
    int position;
};

double simulate_cars(Car cars[], int car_num, int max_iteration, int v_max, double p) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 数据分段
    int offsets[size];
    int lengths[size];
    for (int i = 0; i < size; ++i) {
        offsets[i] = rank * car_num / size;
        lengths[i] = ((i == size - 1) ? car_num : (rank + 1) * car_num / size) - offsets[i];
    }

    int start_index = offsets[rank];
    int end_index = offsets[rank] + lengths[rank];

    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<double> gen(0, 1);

    for (int j = 0; j < max_iteration; ++j) {
        // 每一块区域的队头数据要从前面获取，队尾数据向后发送
        if (rank != size - 1) {
            MPI_Send(cars + end_index - 1, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank != 0) {
            MPI_Status status;
            MPI_Recv(cars + start_index - 1, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        for (int k = start_index; k < end_index; ++k) {
            // 第一辆车 distance = inf
            int distance = k == 0 ? std::numeric_limits<int>::max() : cars[k - 1].position - cars[k].position - 1;
            if (distance > cars[k].velocity) {
                if (cars[k].velocity < v_max) {
                    cars[k].velocity++;
                }
            } else {
                cars[k].velocity = distance;
            }
            if (gen(engine) < p) {
                cars[k].velocity--;
            }
            if (cars[k].velocity < 0) {
                cars[k].velocity = 0;
            }
            cars[k].position += cars[k].velocity;
        }
    }

    // 结束后数据收集到 root 线程
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, lengths[rank], MPI_INT, cars, lengths, offsets, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(cars + offsets[rank], lengths[rank], MPI_INT, cars, lengths, offsets, MPI_INT, 0, MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    return end_time - start_time;
}


int main(int argc, char *argv[]) {

    int car_num = std::stoi(argv[1]);
    int max_iteration = std::stoi(argv[2]);

    double p = 0.2;
    int v_max = 15;
    auto cars = new Car[car_num];

    for (int i = 0; i < car_num; ++i) {
        cars[i].velocity = 0;
        cars[i].position = -i;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "拥堵模拟\n";
        std::cout << "线程数\t车数\t迭代次数\t用时(ms)\n";
    }

    double time = simulate_cars(cars, car_num, max_iteration, v_max, p);

    if (rank == 0) {
        std::cout << size << "\t"
                  << car_num << '\t'
                  << max_iteration << '\t'
                  << std::setprecision(4) << time * 1e3 << '\n';
        // todo 输出到文件
    }

    MPI_Finalize();

    delete[] cars;

    return 0;
}
