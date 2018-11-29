cmake_minimum_required(VERSION 3.12)
project(black_caffe)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_DIR /usr/local/cuda)
set(CUDA_LIB ${CUDA_DIR}/lib64)
set(GLOG_LIB /usr/lib/x86_64-linux-gnu)

include_directories(src)
include_directories(${CUDA_DIR}/include)
include_directories(src/gtest)

aux_source_directory(src/caffeine SRC_LIST)
#aux_source_directory(src/gtest GTEST_SRCS)
set(GTEST_SRCS src/gtest/gtest-all.cpp)
set(TEST_SRC_LIST src/caffeine/test_syncedmem.cpp)

message(STATUS "SORUCE LIST: " ${SRC_LIST})

#add_library(caffeine SHARED ${SRC_LIST})
add_executable(test_caffeine ${SRC_LIST} ${TEST_SRC_LIST} ${GTEST_SRCS})

find_library(CUDART cudart ${CUDA_LIB})
find_library(CUBLAS cublas ${CUDA_LIB})
find_library(GLOG glog ${GLOG_LIB})

#target_link_libraries(caffeine ${CUDART} ${CUBLAS})
target_link_libraries(test_caffeine ${CUDART} ${CUBLAS} ${GLOG} pthread)