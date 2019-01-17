//
// Created by chujie on 12/2/18.
//
#include "caffe/common.hpp"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cstring>

#include "caffe/test/test_caffe_main.hpp"

namespace caffe{
class TestCommon : public ::testing::Test{};

TEST_F(TestCommon, TestCublasHandler){
    int cuda_device_id;
    CUDA_CHECK(cudaGetDevice(&cuda_device_id));
    LOG(INFO) << "Cuda device id: " << cuda_device_id;
    EXPECT_TRUE(Caffe::cublas_handle());
}

TEST_F(TestCommon, TestVslStream) {
        EXPECT_TRUE(Caffe::vsl_stream());
    }

TEST_F(TestCommon, TestBresMode){
    EXPECT_EQ(Caffe::CPU, Caffe::mode());
    Caffe::set_mode(Caffe::GPU);
    EXPECT_EQ(Caffe::GPU, Caffe::mode());
}
}// namespace caffe
