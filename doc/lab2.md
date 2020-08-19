# 并行计算 lab2 —— 蒙特卡洛模拟

实验环境：Ubuntu 20.04, GCC 9.3.0, OpenMP 4.5, MPI 3.1

硬件配置：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, 32 Cores, 64 Threads

## 算法分析与设计

设线程数为 `size`, 每辆车的运动仅和自身状态、前车距离有关，因而可将 `car_num` 长度的组均匀划分为 `size` 个连续段，每个进程负责一段。

计算流程如下：

1. 初始化：使用 `MPI_Scatterv` 把主进程的车辆初始数据分发到每个其他进程;
2. 迭代：
   1. 第 `1` ~ `size-1` 个分段所在的进程向其后一段所在线程发送本段最后一辆车的信息；
   2. 第 `2` ~ `size` 个分段所在进程从前面一段所在线程接受所发送的车辆信息；
   3. 每段独立更新车辆信息；
3. 结束：通过 `MPI_Gatherv` 把所有车辆信息收集到 root 进程。

**可以看出，除了开始和结束阶段，算法并无太多的进程间消息传递，因而程序预计可以获得较好的加速比。**

## 核心代码

定义结构体 `Car`:

```c++
struct Car {
    int velocity;
    int position;
};
```

自定义 MPI 类型 `MPI_CAR`, 便于使用 MPI 的消息传递函数：

```c++
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
```

初始化分发数据：

```c++
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
```

迭代的核心过程
```c++
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
```

最后收集数据到主进程：

```c++
// 结束后数据收集到 root 线程
MPI_Gatherv(cars_local, local_len, MPI_CAR, cars, lengths, offsets, MPI_CAR, 0, MPI_COMM_WORLD);
```

## 实验结果

**所有数字均为重复 20 次取平均值。**

运行时间 (ms)：

| (车辆数，模拟周期数)\进程数 | 1          | 2          | 4          | 8         |
| --------------------------- | ---------- | ---------- | ---------- | --------- |
| (100000, 2000)              | 3184.37486 | 1620.22420 | 822.36850  | 443.96650 |
| (500000, 500)               | 4000.60224 | 2043.15165 | 1028.38714 | 546.40806 |
| (1000000, 300)              | 4804.75453 | 2448.60633 | 1242.85346 | 661.39782 |

加速比：

| (车辆数，模拟周期数)\进程数 | 1   | 2       | 4       | 8       |
| --------------------------- | --- | ------- | ------- | ------- |
| (100000, 2000)              | 1   | 1.96539 | 3.87220 | 7.17255 |
| (500000, 500)               | 1   | 1.95805 | 3.89017 | 7.32163 |
| (1000000, 300)              | 1   | 1.96224 | 3.86590 | 7.26454 |

程序同时会把模拟结束时的车辆数据输出到文件 `lab2_<车辆数>x<迭代次数>.txt`, 内容大致如下（10 辆车迭代 5 轮）：

```
car_num = 10
max_iter = 5
position        velocity
10      3
5       3
1       2
-2      1
-4      0
-5      0
-6      0
-7      0
-8      0
-9      0
```

## 分析与总结

**可以看到，程序如预期一样获得了良好的加速比。**

`v_max=12`, `p=0.2` 时，1000000 辆车迭代 300 次结果如下：

```
car_num = 1000000
max_iter = 300
position	velocity
3446	12
3430	12
3402	12
3388	12
3369	12

......

199	12      <- 第 166 辆
186	11
163	11
145	11
133	12
102	12
82	10
65	8
56	8
44	5
38	7
30	6
14	4
9	3
0	0
-1	0
-2	0
-3	0
-4	1
-6	0
-7	8       <- 第 186 辆
-29	12
-49	11
-68	8
-77	8
-90	7
-111	6
-125	3
-132	3
-136	2
-139	1
-141	0
-142	1
-145	2
-149	0
-150	0

......
```

第 `i` 辆车的初始位置为 `-i`, 可以看到 300 轮迭代之后，前面几辆车已经以稳定的速度开出很远，但从大致 166 辆车开始，后面的车辆速度较低，平均速度不到 2/轮。第 186 辆车附近可以看到明显的车辆疏密变化。

结合实际中的堵车现象，可以看出堵车并非只由道路容纳量过少导致，由于每辆车有一定概率 `p` 随机减速一个单位（现实中则为各种意外导致的车辆减速），后车由于要保持安全距离，也会减速，同时由于车的加速远比减速缓慢，堵车此时就会发生。这种堵车会以类似疏密波的形式在车流中传播（如 186 辆车附近），但从绝对位置来看，堵车点会稳定在一个位置，堵车点不断有车辆离开和加入。
