cmake_minimum_required(VERSION 3.12)
project(black_caffe)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_DIR /usr/local/cuda)
set(CUDA_LIB_DIR ${CUDA_DIR}/lib64)
set(GLOG_LIB_DIR /usr/lib/x86_64-linux-gnu)
set(MKL_LIB_DIR /opt/intel/mkl)

include_directories(src)
include_directories(${CUDA_DIR}/include)
include_directories(src/gtest)
include_directories(/usr/local/include)
include_directories(/opt/intel/mkl/include)
include_directories(src/caffeine/proto)

#find protobuf package
find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
    message(STATUS "protobuf library found")
else()
    message(FATAL_ERROR "protobuf not found")
endif()


PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS src/caffeine/proto/layer_param.proto)
include_directories(${PROTOBUF_INLCUDE_DIRS})

message(STATUS "proto include: ${PROTOBUF_INCLUDE_DIRS}")
message(STATUS "proto source: ${PROTO_SRCS}")
message(STATUS "proto Header: ${PROTO_HDRS}")

aux_source_directory(src/caffeine SRC_LIST)
aux_source_directory(src/caffeine/test TEST_SRCS)
set(GTEST_SRCS src/gtest/gtest-all.cpp)

message(STATUS "source: ${SRC_LIST}")
message(STATUS "test source: ${TEST_SRCS}")
message(STATUS "GTest source: ${GTEST_SRCS}")
message(STATUS "SORUCE LIST:  ${SRC_LIST}")

#add_library(caffeine SHARED ${SRC_LIST})
add_executable(test_caffeine ${PROTO_SRCS} ${PROTO_HDRS} ${SRC_LIST} ${TEST_SRCS} ${GTEST_SRCS})

find_library(CUDART cudart ${CUDA_LIB_DIR})
find_library(CUBLAS cublas ${CUDA_LIB_DIR})
find_library(GLOG glog ${GLOG_LIB_DIR})
find_library(PROTOBUF protobuf ${GLOG_LIB_DIR})
find_library(MKL_RT mkl_rt ${MKL_LIB_DIR})
find_library(MKL_INTEL_THREAD mkl_intel_thread ${MKL_LIB_DIR})
find_library(CURAND curand ${CUDA_LIB_DIR})


#target_link_libraries(caffeine ${CUDART} ${CUBLAS})
target_link_libraries(test_caffeine
        ${CUDART}
        ${CUBLAS}
        ${GLOG}
        ${PROTOBUF}
        ${CURAND}
        ${PROTOBUF_LIBRARIES}
        pthread
        cuda)