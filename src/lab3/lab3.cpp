#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cmath>
#include <cassert>
#include <mpi.h>


const double G = 6.67e-11;

struct Position {
    double x;
    double y;
};

struct Velocity {
    double vx;
    double vy;
};

struct Acceleration {
    double ax;


    double ay;
};

// 计算小球的总受力
inline Acceleration
compute_acceleration(const double masses[], const Position positions[], int self_index, int body_num) {
    Acceleration a{0, 0};
    double self_x = positions[self_index].x;
    double self_y = positions[self_index].y;
    for (int i = 0; i < body_num; i++) {
        if (i != self_index) {
            double dx = positions[i].x - self_x;
            double dy = positions[i].y - self_y;
            double f = G * masses[i] / (dx * dx + dy * dy);
            double distance = std::sqrt(dx * dx + dy * dy);
            a.ax += dx / distance * f;
            a.ay += dy / distance * f;
        }
    }
    return a;
}


double n_body(double masses[], Position positions[], Velocity velocities[],
              int body_num, double total_time, double time_granularity) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建 MPI_POSITION, MPI_VELOCITY
    MPI_Datatype MPI_POSITION, MPI_VELOCITY;
    int element_lens[] = {1, 1};
    MPI_Aint element_offsets[2];
    Position position_tmp{};
    MPI_Get_address(&position_tmp.x, &element_offsets[0]);
    MPI_Get_address(&position_tmp.y, &element_offsets[1]);
    element_offsets[1] -= element_offsets[0];
    element_offsets[0] = 0;
    MPI_Datatype element_types[] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(2, element_lens, element_offsets, element_types, &MPI_POSITION);
    MPI_Type_create_struct(2, element_lens, element_offsets, element_types, &MPI_VELOCITY);
    MPI_Type_commit(&MPI_POSITION);
    MPI_Type_commit(&MPI_VELOCITY);

    // 数据分段
    int offsets[size];
    int lengths[size];
    for (int i = 0; i < size; ++i) {
        offsets[i] = i * body_num / size;
        lengths[i] = (i + 1) * body_num / size - offsets[i];
    }

    int local_len = lengths[rank];

    // mass 数据一次性广播即可
    // 需要传送的是 position 数据
    if (rank != 0) {
        masses = new double[body_num];
        positions = new Position[body_num];
    }
    MPI_Bcast(masses, body_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions, body_num, MPI_POSITION, 0, MPI_COMM_WORLD);

    auto positions_local = positions + offsets[rank]; // 无需单独的 position_local

    // velocity 数据无需传送
    auto velocities_local = new Velocity[local_len];
    MPI_Scatterv(velocities, lengths, offsets, MPI_VELOCITY, velocities_local, local_len, MPI_VELOCITY, 0,
                 MPI_COMM_WORLD);

    // 计算
    for (int t = 0; t < total_time / time_granularity; t++) {
        for (int i = 0; i < local_len; i++) {
            auto a = compute_acceleration(masses, positions, i + offsets[rank], body_num);
            positions_local[i].x += velocities_local[i].vx * time_granularity
                                    + 0.5 * a.ax * time_granularity * time_granularity;
            positions_local[i].y += velocities_local[i].vy * time_granularity
                                    + 0.5 * a.ay * time_granularity * time_granularity;
            velocities_local[i].vx += a.ax * time_granularity;
            velocities_local[i].vy += a.ay * time_granularity;
        }
        // 同步位置
        MPI_Allgatherv(MPI_IN_PLACE, local_len, MPI_POSITION, positions, lengths, offsets, MPI_POSITION,
                       MPI_COMM_WORLD);
    }

    // 收集位置信息
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, local_len, MPI_POSITION, positions, lengths, offsets, MPI_POSITION, 0,
                    MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(positions_local, local_len, MPI_POSITION, positions, lengths, offsets, MPI_POSITION,
                    0,
                    MPI_COMM_WORLD);
    }
    // 收集速度信息
    MPI_Gatherv(velocities_local, local_len, MPI_VELOCITY, velocities, lengths, offsets, MPI_VELOCITY,
                0,
                MPI_COMM_WORLD);

    if (rank != 0) {
        delete[] masses;
        delete[] positions;
    }
    delete[] velocities_local;

    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char *argv[]) {

    int body_num = std::stoi(argv[1]);
    double total_time = std::stod(argv[2]);
    double time_granularity = std::stod(argv[3]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *masses = nullptr;
    Position *positions = nullptr;
    Velocity *velocities = nullptr;

    if (rank == 0) {
        masses = new double[body_num];
        positions = new Position[body_num];
        velocities = new Velocity[body_num];
        for (int i = 0; i < body_num; ++i) {
            masses[i] = 10000;
            velocities[i].vx = 0;
            velocities[i].vy = 0;
        }
        int s = (int) std::sqrt(body_num);
        if (s != std::sqrt(body_num)) {
            std::cerr << "body_num must be squired number" << std::endl;
        }
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                positions[i * s + j].x = i;
                positions[i * s + j].y = j;
            }
        }
    }

    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "n-body 模拟\n";
        std::cout << "线程数\tbody 数\t模拟时间(s)\t用时(ms)\n";
    }

    double time = n_body(masses, positions, velocities, body_num, total_time, time_granularity);

    if (rank == 0) {
        std::cout << size << "\t"
                  << body_num << '\t'
                  << total_time << '\t'
                  << std::setprecision(4) << time * 1e3 << '\n';
        // 输出到文件
        std::ofstream file;
        file.open(std::string("lab3_") + argv[1] + "_" + argv[2] + "_" + argv[3] + ".txt", std::ios::out);
        file << "body_num = " << body_num << '\n';
        file << "total_time = " << total_time << '\n';
        file << "time_granularity = " << time_granularity << '\n';
        file << "mass\tposition\tvelocity" << '\n';
        for (int i = 0; i < body_num; ++i) {
            file << masses[i] << '\t'
                 << '(' << positions[i].x << ',' << positions[i].y << ')' << '\t'
                 << '(' << velocities[i].vx << ',' << velocities[i].vy << ')' << '\n';
        }
        file.flush();
        file.close();
    }

    if (rank == 0) {
        delete[] masses;
        delete[] positions;
        delete[] velocities;
    }

    MPI_Finalize();

    return 0;
}
