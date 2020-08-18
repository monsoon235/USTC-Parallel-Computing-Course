import os
from typing import List

repeat_time: int = 50

thread_num_list = [1, 2, 4, 8, 16, 32]


def execute(cmd: str) -> List[str]:
    r = os.popen(cmd)
    text = r.readlines()
    r.close()
    return text


def lab1_pi():
    max_iter_list = [1000, 10000, 50000, 100000]
    print('lab1 pi omp')
    print('规模\\进程数\t' + '\t'.join(map(str, thread_num_list)))
    for max_iter in max_iter_list:
        print(max_iter, end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'bin/lab1-pi-omp {thread_num} {max_iter}')
                time += float(text[-1].split('\t')[2])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()
    print('lab1 pi mpi')
    print('规模\\进程数\t' + '\t'.join(map(str, thread_num_list)))
    for max_iter in max_iter_list:
        print(max_iter, end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'mpirun -n {thread_num} bin/lab1-pi-mpi {max_iter}')
                time += float(text[-1].split('\t')[2])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()


def lab1_prime():
    max_list = [1000, 10000, 100000, 500000]
    print('lab1 pi omp')
    print('规模\\进程数\t' + '\t'.join(map(str, thread_num_list)))
    for max in max_list:
        print(max, end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'bin/lab1-prime-omp {thread_num} {max}')
                time += float(text[-1].split('\t')[2])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()
    print('lab1 pi mpi')
    print('规模\\进程数\t' + '\t'.join(map(str, thread_num_list)))
    for max in max_list:
        print(max, end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'mpirun -n {thread_num} bin/lab1-prime-mpi {max}')
                time += float(text[-1].split('\t')[2])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()


def lab2():
    print('lab2')
    car_num_list = [100000, 500000, 1000000]
    max_iter_list = [2000, 500, 300]
    print('(车辆数量,周期数)\\进程数' + '\t'.join(map(str, thread_num_list)))
    for car_num, max_iter in zip(car_num_list, max_iter_list):
        print((car_num, max_iter), end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'mpirun -n {thread_num} bin/lab2 {car_num} {max_iter}')
                time += float(text[-1].split('\t')[3])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()


def lab3():
    print('lab3')
    body_num_list = [64, 256]
    total_time = 100
    time_granularity = 0.0001
    print('(小球数,模拟时间,时间粒度)\\进程数' + '\t'.join(map(str, thread_num_list)))
    for body_num in body_num_list:
        print((body_num, total_time, time_granularity), end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'mpirun -n {thread_num} bin/lab3 {body_num} {total_time} {time_granularity}')
                time += float(text[-1].split('\t')[3])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()


def lab4():
    print('lab4')
    arr_len_list = [1e6, 2e6, 5e6, 1e7]
    print('数组长度\\进程数' + '\t'.join(map(str, thread_num_list)))
    for arr_len in arr_len_list:
        print(arr_len, end='\t')
        for thread_num in thread_num_list:
            time = 0
            for _ in range(repeat_time):
                text = execute(f'mpirun -n {thread_num} bin/lab4 {arr_len}')
                time += float(text[-1].split('\t')[2])
            if thread_num == 1:
                time_1_thread = time
            print(f'{time / repeat_time}/{time_1_thread / time}', end='\t')
        print()


if __name__ == '__main__':
    lab1_pi()
    lab1_prime()
    lab2()
    lab3()
    lab4()
