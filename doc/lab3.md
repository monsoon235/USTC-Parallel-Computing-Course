# 并行计算 lab3 —— N 体问题

实验环境：Ubuntu 20.04, GCC 9.3.0, OpenMP 4.5, MPI 3.1

硬件配置：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, 32 Cores, 64 Threads

## 算法分析与设计

本实验要求模拟二维平面上的 N 体问题。

一个质点需要**质量**、**位置**、**速度**三个量来进行描述，计算一个小球的受力则需要其他所有球的质量和位置信息，这两种信息在每轮迭代时都需要全局同步，速度则每个进程维护自己的一部分即可。

设线程数为 `size`，每个进程负责计算数量大致相等的小球的运动，计算的过程大致如下：

1. 初始化阶段：通过 `MPI_Bcast` 把主进程所有的的质量和位置数据广播到全部进程，并通过 `MPI_Scatterv` 发送每个进程所计算小球的初始速度信息；
2. 迭代：
   1. 每个进程分别根据全局质量和位置信息本地小球的加速度（可以直接使用 $a=G\cfrac{m_2}{r^2}$, 而不用计算 $F=G\cfrac{m_1 m_2}{r^2}$ 后再计算 $a=\cfrac{F}{m_1}$）;
   2. 在 $\delta t$ 的时间粒度中，小球的位移为 $\delta \vec{x}=\vec{v} \delta t+\cfrac{1}{2}\vec{a}(\delta t)^2$, 速度变化为 $\delta \vec{v}=\vec{a} \delta t$;
   3. 更新完本地小球的位置和速度信息后，通过 `MPI_Allgatherv` 在进程间同步所有小球的位置信息；
3. 结束：通过 `MPI_Gatherv` 把所有小球的位置和速度信息收集到主进程；

**可以看到，迭代时有大量的进程间通信，进程间通信的速度会对程序的加速比造成影响。**

## 核心代码

**代码中并没有按照实验要求实现 3 个函数，因为拆分出这三个函数并不利于代码结构组织，但其功能都已被正确实现。**

定义三种结构体：

```c++
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
```

计算一个小球受力（加速度）的辅助函数，和 `compute_force()` 的功能接近：

```c++
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
```

自定义 MPI 类型 `MPI_POSITION` 和 `MPI_VELOCITY`：

```c++
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
```

初始化阶段的数据分发：

```c++
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

// velocity 数据无需每次同步
auto velocities_local = new Velocity[local_len];
MPI_Scatterv(velocities, lengths, offsets, MPI_VELOCITY, velocities_local, local_len, MPI_VELOCITY, 0, MPI_COMM_WORLD);
```

迭代过程，包含 `compute_velocities()` 和 `compute_positions()` 的功能：

```c++
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
    MPI_Allgatherv(MPI_IN_PLACE, local_len, MPI_POSITION, positions, lengths, offsets, MPI_POSITION, MPI_COMM_WORLD);
}
```

最后收集数据：

```c++
// 位置信息在循环中已经被同步了，不需要再收集
// 收集速度信息
MPI_Gatherv(velocities_local, local_len, MPI_VELOCITY, velocities, lengths, offsets, MPI_VELOCITY, 0, MPI_COMM_WORLD);
```

## 实验结果

模拟时间为 `10s`, 时间粒度为 `0.0001s`.

**所有数字均为重复 20 次取平均值。**

运行时间 (ms)：

| 小球数\进程数 | 1           | 2           | 4           | 8          |
| ------------- | ----------- | ----------- | ----------- | ---------- |
| 64            | 2758.47410  | 1499.82650  | 1029.72180  | 862.79294  |
| 256           | 44564.35660 | 22726.19126 | 11963.69162 | 6760.62384 |

加速比：

| 小球数\进程数 | 1   | 2       | 4       | 8       |
| ------------- | --- | ------- | ------- | ------- |
| 64            | 1   | 1.83919 | 2.67885 | 3.19714 |
| 256           | 1   | 1.96092 | 3.72496 | 6.59175 |

程序同时会把模拟结束时的小球数据输出到文件 `lab3_<小球数>_<总时间>_<时间粒度>.txt`, 内容大致如下：

```
body_num = 4
total_time = 10
time_granularity = 0.0001
mass	position	velocity
10000	(4.51424e-05,4.51424e-05)	(9.02874e-06,9.02874e-06)
10000	(4.51424e-05,0.999955)	(9.02874e-06,-9.02874e-06)
10000	(0.999955,4.51424e-05)	(-9.02874e-06,9.02874e-06)
10000	(0.999955,0.999955)	(-9.02874e-06,-9.02874e-06)
```

## 分析与总结

可以看到`body_num=256` 时和 `body_num=64` 时加速比都显著小于进程数，由于程序每次迭代都进行了数据同步，这是可以理解的。

同时， `body_num=256` 时的加速比显著高于 `body_num=64`. 一个可能的解释如下：当数据量较少时，MPI 进程通信的时间消耗是大致固定的，主要是系统调用的时间耗费，基本和数据量无关。`body_num=256` 时和 `body_num=64` 时，每次迭代中同步位置的 `MPI_Allgatherv` 传输的数据量都很小，同时前者的计算时间耗费是后者的 4 倍，因而前者会有更高的加速比。
