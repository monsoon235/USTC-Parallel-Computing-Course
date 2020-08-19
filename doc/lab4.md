# 并行计算 lab4 —— PSRS 并行排序

实验环境：Ubuntu 20.04, GCC 9.3.0, OpenMP 4.5, MPI 3.1

硬件配置：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, 32 Cores, 64 Threads

## 算法分析与设计

设总进程数为 `size`;

PSRS 的算法如下：

1. 主进程将数组均匀分段并用把每段发送到对应的进程；
2. 每个进程调用串行排序算法进行局部排序；
3. 每个进程从数组中选取 `size` 个样本，并将这些样本发送到主进程；
4. 主进程对总计 `size*size` 个样本排序，从中均匀选取 `size-1` 个新样本并发送到每个进程；
5. 每个进程根据 `size-1` 个新样本作为分割点，将排序好的局部数组分为 `size` 个部分；
6. 每个进程将第 `i` 个数组部分发送至第 `i` 个进程，于是第 `i` 个进程就有了所有进程的第 `i` 个数组部分；
7. 每个进程再把收到的这些排序好的部分用归并的方法组合为新的有序局部数据；
8. 主进程收集这些新的局部有序数组。

程序的主要通信消耗来自第 6 步，通信的速度会对加速比有一定影响。

## 核心代码

一个多数组归并函数，是第 7 步的辅助函数：

```c++
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
```

第 1 步:

```c++
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
```

第 2~4 步，包括局部排序和采样：

```c++
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
```

第 5 步，局部分段：

```c++
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
```

第 6 步，交换分段，首先需要交换分段长度，分配新的空间，然后再发送分段本身：

```c++
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
```

第 7 步，局部归并：

```c++
// 每个线程重新排序，可以用归并的方法
delete[] local;
local = new int[local_len];
multi_merge(local_new, local_len, offsets, size, local);
delete[] local_new;
```

第 8 步，收集数据，同样需要每个进程发送分段长度：

```c++
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
```

**排序完成后，会调用 `std::is_sorted` 检查排序是否正确，会输出“正确”、“错误”。**

```c++
std::is_sorted(arr, arr + arr_len) ? "正确" : "错误"
```

## 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间 (ms)：

| 数组长度\进程数 | 1          | 2         | 4         | 8         |
| --------------- | ---------- | --------- | --------- | --------- |
| 1000000         | 115.75843  | 69.52046  | 41.17741  | 27.98210  |
| 2000000         | 240.97683  | 143.70351 | 82.16116  | 54.56605  |
| 5000000         | 637.62899  | 377.81044 | 215.74692 | 138.57150 |
| 10000000        | 1343.03522 | 789.66174 | 438.44633 | 282.30548 |

加速比：

| 数组长度\进程数 | 1   | 2       | 4       | 8       |
| --------------- | --- | ------- | ------- | ------- |
| 1000000         | 1   | 1.66509 | 2.81121 | 4.13687 |
| 2000000         | 1   | 1.67690 | 2.93297 | 4.41624 |
| 5000000         | 1   | 1.68769 | 2.95544 | 4.60144 |
| 10000000        | 1   | 1.70077 | 3.06316 | 4.75738 |

程序输出如下：

![](../pic/lab4_result.png)

**由于已经使用了 `std::is_sorted` 检查排序是否正确，不再把数组打印出来肉眼检查。**

## 分析与总结

可以看到各种长度下，算法的加速比都显著低于进程数，体现了进程间通信的影响，特别是第 6 步，会调用 `size` 次 `MPI_Gatherv`，`size` 数量较大时，其通信时间会显著增长。

同时，数组长度越大，相同进程数的加速比越大，可以解释为 MPI 通信时间中有一部分和通信数据量无关的时间占用，因而数组长度越大，这些占用的比例越低，加速比也就越大。
