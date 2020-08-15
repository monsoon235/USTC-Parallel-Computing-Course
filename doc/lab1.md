# 并行计算 lab1

实验环境：Ubuntu 20.04, GCC 9.3.0, OpenMP 4.5, MPI 3.1

硬件配置：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, 32 Cores, 64 Threads

## 计算 $\pi$

## 算法设计与分析

$\pi$ 的近似计算公式为：

$$\pi\approx \frac{1}{n} \sum_{i=0}^n \frac{4}{1+\left(\frac{i+0.5}{n}\right)^2}$$

各项之间完全独立，因而可以并行计算各项再规约。

### MPI

设共有 `size` 个线程，第 `k` 个线程负责计算所有 `i % size == k` 的项之和:

```c++
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

double pi_local = 0;
double pi = 0;
double step = 1.0 / max_iteration;

for (int i = rank; i < max_iteration; i += size) {
    double x = (i + 0.5) * step;
    pi_local += 4.0 / (1.0 + x * x);
}
```

最后再用 `MPI_Reduce` 进行归约求和得到结果：

```c++
MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
pi /= max_iteration;
```

结果如下（用时为重复 20 次取平均值）：

运行时间：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

加速比：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

### OpenMP

OpenMP `parallel for` 制导的语句可以直接把一个循环变成并行块，代码如下：

```c++
    double pi = 0;
    double step = 1.0 / max_iteration; // 减少除法运算

#pragma omp parallel for reduction(+ : pi)
    for (int i = 0; i < max_iteration; i++) {
        double x = (i + 0.5) * step;
        pi += 4.0 / (1 + x * x);
    }

    pi /= max_iteration;
```

结果如下（用时为重复 20 次取平均值）：

运行时间：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

加速比：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

## 统计素数个数

算法的数学原理为筛法：

要选出 $1$ ~ $n$ 之间的素数，先假定 $\geq 2$ 的数都是素数，然后从 $2$ 开始依次遍历每一个数 $i$ 直到 $\sqrt{n}$，如果这个数是素数，则在素数中去除 $i$ 所有的倍数（不包括 $i$ 本身）。最后剩下的则为真正的素数。

### MPI

为了减少线程间的数据传递，最好的做法是，在初始阶段先用串行筛法找出 $1$ ~ $\sqrt{n}$ 的所有素数（否则某一线程修改这一段的数据后，就要进行一次广播），然后每个线程等额划分一段数组，利用已知的 $1$ ~ $\sqrt{n}$ 素数表在分配的数组段中进行筛法，再在此段中统计素数个数，最后用 `MPI_Reduce` 把数据汇总到主线程。

串行筛法：

```c++
// 串行算法，小规模
void mark_prime(bool is_prime[], int max) {
    std::fill_n(is_prime, max + 1, true);
    is_prime[0] = false;
    is_prime[1] = false;
    double stop = std::sqrt(max);
    for (int i = 2; i <= stop; i++) {
        if (is_prime[i]) {
            for (int j = i * 2; j < max; j += i) {
                is_prime[j] = false;
            }
        }
    }
}
```

分段筛法如下：

```c++
// 先找出 1~√max 的素数
int stop = std::sqrt(max);
mark_prime(is_prime, stop);

// 数据分段
int start_index = std::max(rank * (max + 1) / size, stop + 1);
int end_index = (rank + 1) * (max + 1) / size;
for (int i = 2; i <= stop; i++) {
    if (is_prime[i]) {
        // start_index 起第一个 i 的倍数
        int start = (start_index % i == 0) ? start_index : start_index + (i - start_index % i);
        for (int j = std::max(start, 2 * i); j < end_index; j += i) {
            is_prime[j] = false;
        }
    }
}
```

最后统计素数个数，显然可以只统计奇数中的素数个数，再把 `2` 加上，以加快速度：

```c++
// 统计
start_index = (rank * (max + 1) / size) / 2 * 2 + 1;// 只统计奇数
int count = 0;
int count_local = 0;
for (int i = start_index; i < end_index; i += 2) {
    if (is_prime[i]) {
        count_local++;
    }
}
MPI_Reduce(&count_local, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
if (max >= 2) {
    count++;
}
```

结果如下（用时为重复 20 次取平均值）：

运行时间：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

加速比：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 100000      |     |     |     |     |
| 500000      |     |     |     |     |

### OpenMP

OpenMP 的实现方式和 MPI 完全一样，之所以没有使用更方便的 `parallel for` 制导，是考虑到这种实现会导致几个线程同时对同一个数组的随机读写，对于 Cache 不友好，甚至可能导致比串行算法更低的效率。因而和 MPI 一样采用了分段筛法。

核心和 MPI 版本一致：

```c++
    // 先找出 1~√max 的素数
    int stop = std::sqrt(max);
    mark_prime(is_prime, stop);

    // 筛法
#pragma omp parallel
    {
        int rank = omp_get_thread_num(); // 线程 id
        int size = omp_get_num_threads(); // 线程数
        int start_index = std::max(rank * (max + 1) / size, stop + 1);
        int end_index = (rank + 1) * (max + 1) / size;
        for (int i = 2; i <= stop; i++) {
            if (is_prime[i]) {
                // start_index 起第一个 i 的倍数
                int start = (start_index % i == 0) ? start_index : start_index + (i - start_index % i);
                for (int j = std::max(start, 2 * i); j < end_index; j += i) {
                    is_prime[j] = false;
                }
            }
        }
    }

    // 统计
    int count = max >= 2 ? 1 : 0;
#pragma omp parallel reduction(+:count)
    {
        int rank = omp_get_thread_num(); // 线程 id
        int size = omp_get_num_threads(); // 线程数
        int start_index = rank * (max + 1) / size;
        start_index = start_index / 2 * 2 + 1; // 只统计奇数
        int end_index = (rank + 1) * (max + 1) / size;
        for (int j = start_index; j < end_index; j += 2) {
            if (is_prime[j]) {
                count++;
            }
        }
    }
```

结果如下（用时为重复 20 次取平均值）：

运行时间：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 50000       |     |     |     |     |
| 100000      |     |     |     |     |

加速比：

| 规模\进程数 | 1   | 2   | 4   | 8   |
| ----------- | --- | --- | --- | --- |
| 1000        |     |     |     |     |
| 10000       |     |     |     |     |
| 100000      |     |     |     |     |
| 500000      |     |     |     |     |

## 分析与总结
