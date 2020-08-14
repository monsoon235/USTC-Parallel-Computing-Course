cmake_minimum_required(VERSION 3.14)
project(USTC_Parallel_Computing_Course)

set(CMAKE_CXX_STANDARD 17)

add_executable(lab1-pi-mpi src/lab1/pi-mpi.cpp)
add_executable(lab1-pi-omp src/lab1/pi-omp.cpp)
add_executable(lab1-prime-mpi src/lab1/prime-mpi.cpp)
add_executable(lab1-prime-omp src/lab1/prime-omp.cpp)
add_executable(lab2 src/lab2/lab2.cpp)
add_executable(lab3 src/lab3/lab3.cpp)
add_executable(lab4 src/lab4/lab4.cpp)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(lab1-pi-mpi PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab1-pi-omp PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab1-prime-mpi PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab1-prime-omp PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab2 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab3 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_link_libraries(lab4 PUBLIC OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
