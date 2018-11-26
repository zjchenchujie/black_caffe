cmake_minimum_required(VERSION 3.12)
project(black_caffe)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_DIR /usr/local/cuda)
set(CUDA_LIB ${CUDA_DIR}/lib64)

include_directories(src)
include_directories(${CUDA_DIR}/include)

aux_source_directory(src/caffeine SRC_LIST)
message(STATUS "SORUCE LIST: " ${SRC_LIST})

add_library(caffeine SHARED ${SRC_LIST})

find_library(CUDART cudart ${CUDA_LIB})
find_library(CUBLAS cublas ${CUDA_LIB})

target_link_libraries(caffeine ${CUDART} ${CUBLAS})