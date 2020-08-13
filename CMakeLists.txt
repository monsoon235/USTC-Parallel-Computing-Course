cmake_minimum_required(VERSION 3.14)
project(USTC_Parallel_Computing_Course)

set(CMAKE_CXX_STANDARD 20)

add_executable(exp1 lab1.cpp)
add_executable(exp2 lab2.cpp)
add_executable(exp3 lab3.cpp)
add_executable(exp4 lab4.cpp)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(exp1 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(exp2 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(exp3 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(exp4 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
