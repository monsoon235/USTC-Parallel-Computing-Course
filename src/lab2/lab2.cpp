#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <random>
#include <fstream>
#include <mpi.h>


struct Car {
    int velocity;
    int position;
};

double simulate_cars(Car cars[], int car_num, int max_iter, int v_max, double p) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 自定义 Car 类型
    MPI_Datatype MPI_CAR;
    int element_lens[] = {1, 1};
    MPI_Aint element_offsets[2];
    Car car_tmp{};
    MPI_Get_address(&car_tmp.velocity, &element_offsets[0]);
    MPI_Get_address(&car_tmp.position, &element_offsets[1]);
    element_offsets[1] -= element_offsets[0];
    element_offsets[0] = 0;
    MPI_Datatype element_types[] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, element_lens, element_offsets, element_types, &MPI_CAR);
    MPI_Type_commit(&MPI_CAR);

    // 数据分段
    int offsets[size];
    int lengths[size];
    for (int i = 0; i < size; ++i) {
        offsets[i] = i * car_num / size;
        lengths[i] = (i + 1) * car_num / size - offsets[i];
    }

    auto cars_local = (new Car[lengths[rank] + 1]) + 1; // 空一个位置放前置 car
    int local_len = lengths[rank];

    // 主线程分发数据
    MPI_Scatterv(cars, lengths, offsets, MPI_CAR, cars_local, local_len, MPI_CAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cars_local[-1] = Car{0, std::numeric_limits<int>::max()}; // 第一辆车前面的虚拟无穷远处的 car
    }

    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<double> gen(0, 1);

    for (int i = 0; i < max_iter; i++) {
        // 段之间交流首尾车辆信息
        if (rank != size - 1) {
            MPI_Send(cars_local + local_len - 1, 1, MPI_CAR, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank != 0) {
            MPI_Status status;
            MPI_Recv(cars_local - 1, 1, MPI_CAR, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        // 更新速度
        for (int j = 0; j < local_len; j++) {
            int distance = cars_local[j - 1].position - cars_local[j].position - 1;
            if (distance > cars_local[j].velocity) {
                if (cars_local[j].velocity < v_max) {
                    cars_local[j].velocity++;
                }
            } else {
                cars_local[j].velocity = distance;
            }
            if (gen(engine) < p) {
                cars_local[j].velocity--;
            }
            if (cars_local[j].velocity < 0) {
                cars_local[j].velocity = 0;
            }
        }
        // 更新距离
        for (int j = 0; j < local_len; ++j) {
            cars_local[j].position += cars_local[j].velocity;
        }
    }

    // 结束后数据收集到 root 线程
    MPI_Gatherv(cars_local, local_len, MPI_CAR, cars, lengths, offsets, MPI_CAR, 0, MPI_COMM_WORLD);

    delete[] (cars_local - 1);

    double end_time = MPI_Wtime();
    return end_time - start_time;
}


int main(int argc, char *argv[]) {

    int car_num = std::stoi(argv[1]);
    int max_iter = std::stoi(argv[2]);

    double p = 0.2;
    int v_max = 12;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Car *cars = nullptr;

    if (rank == 0) {
        cars = new Car[car_num];
        for (int i = 0; i < car_num; ++i) {
            cars[i].velocity = 0;
            cars[i].position = -i;
        }
    }

    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "拥堵模拟\n";
        std::cout << "线程数\t车数\t迭代次数\t用时(ms)\n";
    }

    double time = simulate_cars(cars, car_num, max_iter, v_max, p);

    if (rank == 0) {
        std::cout << size << "\t"
                  << car_num << '\t'
                  << max_iter << '\t'
                  << std::setprecision(4) << time * 1e3 << '\n';
        // 输出到文件
        std::ofstream file;
        file.open(std::string("lab2_") + argv[1] + "x" + argv[2] + ".txt", std::ios::out);
        file << "car_num = " << car_num << '\n';
        file << "max_iter = " << max_iter << '\n';
        file << "position\tvelocity" << '\n';
        for (int i = 0; i < car_num; ++i) {
            file << cars[i].position << '\t' << cars[i].velocity << '\n';
        }
        file.flush();
        file.close();
    }

    if (rank == 0) {
        delete[] cars;
    }

    MPI_Finalize();

    return 0;
}
