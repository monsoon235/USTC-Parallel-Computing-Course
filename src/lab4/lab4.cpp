#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <random>
#include <mpi.h>

// 归并
void multi_merge(int arr[], int arr_len, int offsets[], int size, int output[]) {
    int *starts[size];
    int *ends[size];
    for (int i = 0; i < size; ++i) {
        starts[i] = arr + offsets[i];
    }
    ends[size - 1] = arr + arr_len;
    for (int i = 0; i < size - 1; ++i) {
        ends[i] = starts[i + 1];
    }
    for (int i = 0; i < arr_len; ++i) {
        int min = 0;
        while (starts[min] >= ends[min]) {
            min++;
        }
        for (int j = min + 1; j < size; ++j) {
            if (starts[j] < ends[j] && *starts[j] < *starts[min]) {
                min = j;
            }
        }
        output[i] = *starts[min];
        starts[min]++;
    }
}

double PSRS(int arr[], int arr_len) {
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 数据分段
    int offsets[size];
    int lengths[size];
    for (int i = 0; i < size; ++i) {
        offsets[i] = i * arr_len / size;
        lengths[i] = (i + 1) * arr_len / size - offsets[i];
    }

    // 给每个线程分配数据
    auto local = new int[lengths[rank]];
    auto local_len = lengths[rank];
    MPI_Scatterv(arr, lengths, offsets, MPI_INT, local, local_len, MPI_INT, 0, MPI_COMM_WORLD);

    // 区域排序
    std::sort(local, local + local_len);

    // 采样
    int sample[size];
    std::sample(local, local + local_len, sample, size, std::mt19937{std::random_device{}()});

    // 选取划分元素
    int sample_all[size * size];
    int fake_pivot[size + 1];// 添加哨兵方便进行划分
    int *pivot = fake_pivot + 1;
    MPI_Gather(sample, size, MPI_INT, sample_all, size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::sort(sample_all, sample_all + size * size);
        // 此处需要均匀划分
        for (int i = 0; i < size - 1; ++i) {
            pivot[i] = sample_all[(i + 1) * size];
        }
    }
    MPI_Bcast(pivot, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 按 pivot 对局部进行划分
    pivot[-1] = local[0];
    pivot[size - 1] = local[local_len - 1] + 1;

    int part_offset[size]; // 每一部分的开始位置
    int part_len[size];  // 每一部分的长度

    part_offset[0] = 0;
    int now_index = 0;
    for (int i = 0; i < size; ++i) {
        while (now_index < local_len && pivot[i - 1] <= local[now_index] && local[now_index] < pivot[i]) {
            now_index++;
        }
        part_len[i] = now_index - part_offset[i];
        if (i + 1 < size) {
            part_offset[i + 1] = now_index;
        }
    }

    // 发送每个划分长度
    for (int i = 0; i < size; ++i) {
        MPI_Gather(&part_len[i], 1, MPI_INT, lengths, 1, MPI_INT, i, MPI_COMM_WORLD);
    }
    offsets[0] = 0;
    for (int i = 1; i < size; ++i) {
        offsets[i] = offsets[i - 1] + lengths[i - 1];
    }

    local_len = std::accumulate(lengths, lengths + size, 0);
    auto local_new = new int[local_len];

    // 把每个线程的第 i 个划分发送到第 i 个线程
    for (int i = 0; i < size; ++i) {
        MPI_Gatherv(local + part_offset[i], part_len[i], MPI_INT, local_new, lengths, offsets, MPI_INT, i,
                    MPI_COMM_WORLD);
    }

    // 每个线程重新排序，可以用归并的方法
    delete[] local;
    local = new int[local_len];
    multi_merge(local_new, local_len, offsets, size, local);
    delete[] local_new;

    // 所有长度数据汇总到 root
    MPI_Gather(&local_len, 1, MPI_INT, lengths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        offsets[0] = 0;
        for (int i = 1; i < size; ++i) {
            offsets[i] = offsets[i - 1] + lengths[i - 1];
        }
    }

    // 所有数据汇总到 root
    MPI_Gatherv(local, local_len, MPI_INT, arr, lengths, offsets, MPI_INT, 0, MPI_COMM_WORLD);

    delete[] local;

    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char *argv[]) {

    auto arr_len = std::stoi(argv[1]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *arr = nullptr;
    if (rank == 0) {
        arr = new int[arr_len];
        // 初始化
        std::generate_n(arr, arr_len, []() {
            static int i = 0;
            return i++;
        });
        // 打乱
        std::shuffle(arr, arr + arr_len, std::mt19937(std::random_device()()));
    }

    if (rank == 0) {
        std::cout << std::setiosflags(std::ios::fixed);
        std::cout << "PSRS 排序\n";
        std::cout << "线程数\t数组长度\t用时(ms)\t结果检查\n";
    }

    auto time = PSRS(arr, arr_len);

    if (rank == 0) {
        std::cout << size << '\t'
                  << arr_len << '\t'
                  << std::setprecision(4) << time * 1e3 << '\t'
                  << (std::is_sorted(arr, arr + arr_len) ? "正确" : "错误") << std::endl;
    }

    MPI_Finalize();
    return 0;
}

