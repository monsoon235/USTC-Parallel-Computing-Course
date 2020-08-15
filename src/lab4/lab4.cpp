#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <limits>
#include <random>
#include <mpi.h>

using namespace std;

// 将 partition[n] 中的数据按 start[size] 划分后归并到 merged[n] 中
void Multimerge(const int partition[], const int start[], int size, int merged[], int n) {
    int index[size];
    for (int i = 0; i < size; i++)
        index[i] = start[i]; // 初始化指针

    for (int i = 0; i < n; i++) {
        int p = 0; // 首个活跃（未达分段末尾）的指针
        while (p < size && index[p] >= start[p + 1])
            p++;
        if (p >= size)
            break;

        for (int q = p + 1; q < size; q++) // 向后遍历指针
            if (index[q] < start[q + 1])   // 指针活跃
                if (partition[index[p]] > partition[index[q]])
                    p = q; // 取未合并分段的最小者
        merged[i] = partition[index[p]++];
    }
}

int main2(int argc, char *argv[]) {
    int rank, size;
    const int n = std::stoi(argv[1]); // 获取数组长度

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned) (time(nullptr) + rank)); // 随机种子

    double startTime = MPI_Wtime();

    int recvcounts[size];
    int displs[size + 1];

    for (int i = 0; i < size; i++) {
        displs[i] = i * n / size;                                               // 局部队首
        recvcounts[i] = ((i == size - 1) ? n : (i + 1) * n / size) - displs[i]; // 局部长度
    }

    int localLength = recvcounts[rank]; // 局部长度

    int *global = new int[n];
    int *local = new int[localLength];

    /* ======================== Phase I ======================== */
    // Initialization

    if (rank == 0) {
        generate_n(global, n, []() {
            static int i = 1;
            return i++;
        });                                     // 生成 1 ~ n 的序列
        std::shuffle(&global[0], &global[n], std::mt19937(std::random_device()())); // 随机打乱序列

        // 输出乱序序列
        if (n <= 1000) {
            for (int i = 0; i < n; i++)
                cout << setw(3) << global[i] << ((i % 30 == 29) ? '\n' : ' ');
            cout << endl;
        }
    }

    /* ======================== Phase II ======================== */
    // Scatter data, local sort and regular samples collected

    MPI_Scatterv(global, recvcounts, displs, MPI_INT, local, localLength, MPI_INT, 0,
                 MPI_COMM_WORLD); // 将全局数据均匀散播到其它处理器中

    sort(local, local + localLength); // 局部排序

    int sample[size]; // 样本，数量与进程数相同（假设局部长度大于进程总数）

    for (int i = 0; i < size; i++)
        sample[i] = local[i * localLength / size]; // 均匀采样

    int globalSample[size * size];

    MPI_Gather(sample, size, MPI_INT, globalSample, size, MPI_INT, 0, MPI_COMM_WORLD); // 收集全局共计 p^2 个样本

    /* ======================= Phase III ======================= */
    // Gather and merge samples, choose and broadcast p-1 pivots

    int *pivot = (new int[size + 1]) + 1; // 划分元素，特殊的地址设置是为 Phase IV 中越界下标之便

    if (rank == 0) // 通过 multimerge 合并各局部样本
    {
        int start[size]; // 局部样本指针
        int mergedGlobalSample[size * size];

        for (int i = 0; i < size; i++)
            start[i] = i * size; // 从各局部的 0 （相对）开始

        Multimerge(globalSample, start, size, mergedGlobalSample, size * size);

        for (int i = 0; i < size - 1; i++)
            pivot[i] = mergedGlobalSample[i * (size * size) / (size - 1)]; // 划分元素
    }

    MPI_Bcast(pivot, size - 1, MPI_INT, 0, MPI_COMM_WORLD); // 广播划分元素

    /* ======================== Phase IV ======================== */
    // Local data is partitioned

    pivot[-1] = local[0] - 1; // 越界下标，便于统一处理上下界
    pivot[size - 1] = local[localLength - 1];

    int part[size + 1] = {0}; // 按 pivot 对局部进行划分

    for (int k = 0, i = 0; k < size; k++) {
        while ((i < localLength) && (pivot[k - 1] < local[i]) && (local[i] <= pivot[k]))
            i++;
        part[k + 1] = i;
    }

    delete[](pivot - 1);

    int partLength[size];
    for (int i = 0; i < size; i++)
        partLength[i] = part[i + 1] - part[i];

    /* ======================== Phase V ======================== */
    // All i^th classes are gathered and merged

    for (int i = 0; i < size; i++)
        MPI_Gather(partLength + i, 1, MPI_INT, recvcounts, 1, MPI_INT, i, MPI_COMM_WORLD);

    localLength = accumulate(&recvcounts[0], &recvcounts[size], 0); // 重新分配后，局部的总长度
    int *local_reallocated = new int[localLength];                  // 重新分配后的局部段

    displs[0] = 0;
    for (int i = 0; i < size; i++)
        displs[i + 1] = displs[i] + recvcounts[i];

    for (int i = 0; i < size; i++)
        MPI_Gatherv(local + part[i], partLength[i], MPI_INT, local_reallocated, recvcounts, displs, MPI_INT, i,
                    MPI_COMM_WORLD); // 将每个线程的第 i 部分重新分配到第 i 个线程

    delete[] local;

    // 对重新分配的数据 multimerge

    int *mergedLocal = new int[localLength];

    Multimerge(local_reallocated, displs, size, mergedLocal, localLength);

    delete[] local_reallocated;

    /* ======================== Phase VI ======================== */
    // Root processor collects all the data

    MPI_Gather(&localLength, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD); // 收集局部长度

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 0; i < size - 1; i++)
            displs[i + 1] = displs[i] + recvcounts[i];
    }
    MPI_Gatherv(mergedLocal, localLength, MPI_INT, global, recvcounts, displs, MPI_INT, 0,
                MPI_COMM_WORLD); // 收集全局数据

    delete[] mergedLocal;

    /* ========================================================= */

    double endTime = MPI_Wtime();

    // 输出
    if ((rank == 0) && (n <= 1000)) {
        cout << "Sorted: " << endl;
        for (int i = 0; i < n; i++)
            cout << setw(3) << global[i] << ((i % 30 == 29) ? '\n' : ' ');
        cout << endl;
    }

    delete[] global;

    if (rank == 0)
        cout << "Elapsed time = " << endTime - startTime << "s" << endl;

    MPI_Finalize();
    return 0;
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

    int start_index = offsets[rank];
    int end_index = offsets[rank] + lengths[rank];

    auto local = arr + offsets[rank];
    auto local_len = lengths[rank];

    // 区域排序
    std::sort(local, local + local_len);

    // 采样
    auto sample = new int[size];
    std::sample(local, local + local_len, sample, size, std::mt19937{std::random_device{}()});

    // 选取划分元素
    auto sample_all = new int[size * size];
    auto pivot = (new int[size + 1]) + 1;
    MPI_Gather(sample, size, MPI_INT, sample_all, size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::sort(sample_all, sample_all + size * size);
        std::sample(sample_all, sample_all + size * size, pivot, size - 1, std::mt19937{std::random_device{}()});
    }
    MPI_Bcast(pivot, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 按 pivot 对局部进行划分
    pivot[-1] = local[0] - 1;
    pivot[size - 1] = local[local_len - 1];

    // todo

    delete[] sample;
    delete[] sample_all;
    delete[] (pivot - 1);

    double end_time = MPI_Wtime();
    return end_time - start_time;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int arr[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    PSRS(arr, 9);
    MPI_Finalize();
    return 0;
}

