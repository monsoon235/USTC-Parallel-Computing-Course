#include <iostream>
#include <iomanip>
#include <string>
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

struct Force {
    double fx;
    double fy;
};

// 计算小球的总受力
inline Force
compute_force(const double masses[], const Position positions[], const int self_index, const int body_num) {
    Force force{0, 0};
    double self_mass = masses[self_index];
    double self_x = positions[self_index].x;
    double self_y = positions[self_index].y;
    for (int i = 0; i < self_index; ++i) {
        double dx = positions[i].x - self_x;
        double dy = positions[i].y - self_y;
        double f = G * self_mass * masses[i] / (dx * dx + dy * dy);
        double distance = std::sqrt(dx * dx + dy * dy);
        force.fx += dx / distance * f;
        force.fy += dy / distance * f;
    }
    for (int i = self_index + 1; i < body_num; ++i) {
        double dx = positions[i].x - self_x;
        double dy = positions[i].y - self_y;
        double f = G * self_mass * masses[i] / (dx * dx + dy * dy);
        double distance = std::sqrt(dx * dx + dy * dy);
        force.fx += dx / distance * f;
        force.fy += dy / distance * f;
    }
    return force;
}


double n_body(const double masses[], Position positions[], Velocity velocities[],
              int body_num, double total_time, double time_granularity) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype MPI_POSITION, MPI_VELOCITY;
    int element_lens[] = {1, 1};
    MPI_Aint element_offsets[2];
    MPI_Get_address(&positions[0].x, &element_offsets[0]);
    MPI_Get_address(&positions[1].y, &element_offsets[1]);
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
        offsets[i] = rank * body_num / size;
        lengths[i] = ((i == size - 1) ? body_num : (rank + 1) * body_num / size) - offsets[i];
    }

    int start_index = offsets[rank];
    int end_index = offsets[rank] + lengths[rank];

    for (int t = 0; t < total_time / time_granularity; t++) {
        for (int j = start_index; j < end_index; ++j) {
            auto force = compute_force(masses, positions, j, body_num);
            velocities[j].vx += force.fx / masses[j] * time_granularity;
            velocities[j].vy += force.fy / masses[j] * time_granularity;
            positions[j].x += velocities[j].vx * time_granularity;
            positions[j].y += velocities[j].vy * time_granularity;
        }
        // 同步状态
        MPI_Allgatherv(MPI_IN_PLACE, lengths[rank], MPI_POSITION, positions, lengths, offsets, MPI_POSITION,
                       MPI_COMM_WORLD);
        // 速度无需同步
//        MPI_Allgatherv(MPI_IN_PLACE, lengths[rank], MPI_VELOCITY, velocities, lengths, offsets,
//                       MPI_VELOCITY,
//                       MPI_COMM_WORLD);
    }

    // 收集结果
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, lengths[rank], MPI_POSITION, positions, lengths, offsets, MPI_POSITION, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(MPI_IN_PLACE, lengths[rank], MPI_VELOCITY, velocities, lengths, offsets, MPI_VELOCITY, 0,
                    MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(positions, lengths[rank], MPI_POSITION, positions, lengths, offsets, MPI_POSITION, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(velocities, lengths[rank], MPI_VELOCITY, velocities, lengths, offsets, MPI_VELOCITY, 0,
                    MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char *argv[]) {

    int body_num = std::stoi(argv[1]);
    double total_time = std::stod(argv[2]);
    double time_granularity = std::stod(argv[3]);

    int s = (int) std::sqrt(body_num);

    assert(s == std::sqrt(body_num));

    auto masses = new double[body_num];
    auto positions = new Position[body_num];
    auto velocities = new Velocity[body_num];

    for (int i = 0; i < body_num; ++i) {
        masses[i] = 10000;
        velocities[i].vx = 0;
        velocities[i].vy = 0;
    }

    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            positions[i * s + j].x = i;
            positions[i * s + j].y = j;
        }
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        // todo 输出到文件
    }

    MPI_Finalize();

    delete[] masses;
    delete[] positions;
    delete[] velocities;

    return 0;
}
