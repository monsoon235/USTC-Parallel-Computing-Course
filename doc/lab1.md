# 并行计算 lab1 —— 计算 $\pi$ 与素数统计

实验环境：Ubuntu 20.04, GCC 9.3.0, OpenMP 4.5, MPI 3.1

硬件配置：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, 32 Cores, 64 Threads

## 计算 $\pi$

### 算法分析与设计

$\pi$ 的近似计算公式为：

$$\pi\approx \frac{1}{n} \sum_{i=0}^n \frac{4}{1+\left(\frac{i+0.5}{n}\right)^2}$$

各项之间完全独立，因而可以并行计算各项再规约。

### MPI 实现

#### 核心代码

设共有 `size` 个线程，第 `k` 个线程负责计算所有 `i % size == k` 的项之和:

```c++
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
```

最后再用 `MPI_Reduce` 进行归约求和得到结果：

```c++
MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
pi /= max_iter;
```

#### 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间 (ms)：

| 规模\进程数 | 1       | 2       | 4       | 8       |
| ----------- | ------- | ------- | ------- | ------- |
| 1000        | 0.00435 | 0.03031 | 0.05293 | 0.09988 |
| 10000       | 0.02572 | 0.05123 | 0.06188 | 0.08002 |
| 50000       | 0.10790 | 0.10123 | 0.08845 | 0.08201 |
| 100000      | 0.23151 | 0.15766 | 0.12461 | 0.12001 |

加速比：

| 规模\进程数 | 1   | 2       | 4       | 8       |
| ----------- | --- | ------- | ------- | ------- |
| 1000        | 1   | 0.14368 | 0.08227 | 0.04360 |
| 10000       | 1   | 0.50200 | 0.41560 | 0.32139 |
| 50000       | 1   | 1.06588 | 1.21982 | 1.31561 |
| 100000      | 1   | 1.46841 | 1.85780 | 1.92892 |

### OpenMP 实现

#### 核心代码

OpenMP `parallel for` 制导的语句可以直接把一个循环变成并行块，代码如下：

```c++
double pi = 0;
double step = 1.0 / max_iter; // 减少除法运算

#pragma omp parallel for reduction(+ : pi)
for (int i = 0; i < max_iter; i++) {
    double x = (i + 0.5) * step;
    pi += 4.0 / (1 + x * x);
    }

pi /= max_iter;
```

#### 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间：

| 规模\进程数 | 1       | 2       | 4       | 8       |
| ----------- | ------- | ------- | ------- | ------- |
| 1000        | 0.00653 | 0.08484 | 0.12668 | 0.24404 |
| 10000       | 0.02166 | 0.09461 | 0.14773 | 0.24896 |
| 50000       | 0.09281 | 0.12905 | 0.16996 | 0.25573 |
| 100000      | 0.18359 | 0.17240 | 0.19502 | 0.27707 |

加速比：

| 规模\进程数 | 1   | 2       | 4       | 8       |
| ----------- | --- | ------- | ------- | ------- |
| 1000        | 1   | 0.07690 | 0.05150 | 0.02673 |
| 10000       | 1   | 0.22895 | 0.14662 | 0.08700 |
| 50000       | 1   | 0.71913 | 0.54604 | 0.36290 |
| 100000      | 1   | 1.06487 | 0.94136 | 0.66260 |

### 分析与总结



## 统计素数个数

### 算法分析与设计

算法的数学原理为筛法：

要选出 $1$ ~ $n$ 之间的素数，先假定 $\geq 2$ 的数都是素数，然后从 $2$ 开始依次遍历每一个数 $i$ 直到 $\sqrt{n}$，如果这个数是素数，则在素数中去除 $i$ 所有的倍数（不包括 $i$ 本身）。最后剩下的则为真正的素数。

### MPI 实现

#### 核心代码

为了减少线程间的数据传递，最好的做法是，在初始阶段先用串行筛法找出 $1$ ~ $\sqrt{n}$ 的所有素数（否则某一线程修改这一段的数据后，就要进行一次广播），然后每个线程等额划分一段数组，利用已知的 $1$ ~ $\sqrt{n}$ 素数表在分配的数组段中进行筛法，再在此段中统计素数个数，最后用 `MPI_Reduce` 把数据汇总到主线程。

串行筛法：

```c++
// 串行算法，小规模
void mark_prime(bool is_prime[], int max) {
    std::fill_n(is_prime, max + 1, true);
    is_prime[0] = false;
    if (max > 1) {
        is_prime[1] = false;
    }
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
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int offset = rank * (max + 1) / size;
int length = (rank + 1) * (max + 1) / size - offset;

// 先找出 1~√max 的素数
int max_sqrt = std::sqrt(max);
auto is_prime_before_sqrt = new bool[max_sqrt + 1];
std::fill_n(is_prime_before_sqrt, max_sqrt + 1, true);
mark_prime(is_prime_before_sqrt, max_sqrt);

auto is_prime = new bool[length];  // 每个线程负责 [offset, offset+length) 的部分
std::fill_n(is_prime, length, true);
// 去除 0 1
if (offset <= 0 && offset + length > 0) {
    is_prime[0 - offset] = false;
}
if (offset <= 1 && offset + length > 1) {
    is_prime[1 - offset] = false;
}

// 筛法
for (int i = 2; i <= max_sqrt; i++) {
    if (is_prime_before_sqrt[i]) {
        int start = (offset % i == 0) ? 0 : (i - offset % i);
        int real_start = std::max(start, 2 * i - offset);
        for (int j = real_start; j < length; j += i) {
            is_prime[j] = false;
        }
    }
}
```

最后统计素数个数，显然可以只统计奇数中的素数个数，再把 `2` 加上，以加快速度：

```c++
// 统计，只统计奇数
int count_local = 0;
int start = 1 - offset % 2;
for (int i = start; i < length; i += 2) {
    if (is_prime[i]) {
        count_local++;
    }
}
int count = 0;
MPI_Reduce(&count_local, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
if (max >= 2) {
    count++;
}
```

#### 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间：

| 规模\进程数 | 1       | 2       | 4       | 8       |
| ----------- | ------- | ------- | ------- | ------- |
| 1000        | 0.00900 | 0.04406 | 0.06835 | 0.09703 |
| 10000       | 0.05339 | 0.05332 | 0.07820 | 0.08963 |
| 100000      | 0.40853 | 0.24498 | 0.15535 | 0.12657 |
| 500000      | 1.95603 | 1.04064 | 0.55781 | 0.33792 |

加速比：

| 规模\进程数 | 1   | 2       | 4       | 8       |
| ----------- | --- | ------- | ------- | ------- |
| 1000        | 1   | 0.20433 | 0.13173 | 0.09280 |
| 10000       | 1   | 1.00140 | 0.68271 | 0.59572 |
| 100000      | 1   | 1.66762 | 2.62968 | 3.22748 |
| 500000      | 1   | 1.87963 | 3.50659 | 5.78826 |

### OpenMP 实现

#### 核心代码

OpenMP 版本的实现方式和 MPI 版本完全一样，之所以没有使用更方便的 `parallel for` 制导，是考虑到这种实现会导致几个线程同时对同一个数组的随机读写，对于 Cache 不友好，甚至可能导致比串行算法更低的效率。因而和 MPI 一样采用了分段筛法。

核心和 MPI 版本一致：

```c++
auto is_prime = new bool[max + 1];
std::fill_n(is_prime, max + 1, true);

// 先找出 1~√max 的素数
int max_sqrt = std::sqrt(max);
mark_prime(is_prime, max_sqrt);

    // 筛法
#pragma omp parallel
{
    int rank = omp_get_thread_num(); // 线程 id
    int size = omp_get_num_threads(); // 线程数
    int start_index = std::max(rank * (max + 1) / size, max_sqrt + 1);
    int end_index = (rank + 1) * (max + 1) / size;
    for (int i = 2; i <= max_sqrt; i++) {
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

#### 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间：

| 规模\进程数 | 1       | 2       | 4       | 8       |
| ----------- | ------- | ------- | ------- | ------- |
| 1000        | 0.00831 | 0.09038 | 0.14611 | 0.25960 |
| 10000       | 0.02231 | 0.10141 | 0.15509 | 0.25897 |
| 100000      | 0.22529 | 0.21576 | 0.22868 | 0.31719 |
| 500000      | 1.08960 | 0.80084 | 0.66263 | 0.69844 |

加速比：

| 规模\进程数 | 1   | 2       | 4       | 8       |
| ----------- | --- | ------- | ------- | ------- |
| 1000        | 1   | 0.09188 | 0.05683 | 0.03199 |
| 10000       | 1   | 0.21993 | 0.14381 | 0.08612 |
| 100000      | 1   | 1.04412 | 0.98513 | 0.71024 |
| 500000      | 1   | 1.36055 | 1.64434 | 1.56004 |

## 分析与总结

### 计算 $\pi$

可以看出 MPI 实现版本和 OpenMP 实现版本在迭代次数有限（10000 次以内）的情况下，加速比甚至都远小于 1，甚至出现了并行规模越大，耗时约长的现象, 这是因为每个进程的计算很快被完成，反而进程间通信（和并行规模相关）占了主要时间消耗。

同样可以证明这一点的是，随着迭代次数增加，相同进程数情况下的加速比也在增加，因为 `Reduce` 的进程间通信时间和迭代次数无关，耗费在计算上的时间越多，加速比也就越大。

同样可以发现，MPI 实现的版本无论速度还是加速比都优于 OpenMP 实现的版本，猜测原因如下;

1. OpenMP 版本中，fork-join 的并行方式下，并行的线程是在运行中创建的，创建时间也被计入了耗时，但 MPI 版本中，进程的创建是在执行 `mpirun` 命令时就完成了（多个进程被同时创建），不计入耗时，因而 MPI 版本速度更快；
2. OpenMP 版本中 `#pragma omp parallel for reduction(+ : pi)` 的并行方式并不能很好地进行线程调度最大化计算资源使用，因为每次 `for` 循环时间很短，线程池的任务切换太过繁忙；MPI 版本中则无需进行线程调度。

### 统计素数

同样可以看到在迭代次数有限（100000 次以内）的情况下，加速比远小于 1，原因同计算 $\pi$ 的情况。

同样 MPI 版本的速度和加速比都优于 OpenMP 版本，原因同计算 $\pi$ 的情况。
