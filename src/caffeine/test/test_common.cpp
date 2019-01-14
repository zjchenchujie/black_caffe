//
// Created by chujie on 12/2/18.
//
#include "caffeine/common.hpp"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cstring>

#include "caffeine/test/test_caffeine_main.hpp"

namespace caffeine{
class TestCommon : public ::testing::Test{};

TEST_F(TestCommon, TestCublasHandler){
    int cuda_device_id;
    CUDA_CHECK(cudaGetDevice(&cuda_device_id));
    LOG(INFO) << "Cuda device id: " << cuda_device_id;
    EXPECT_TRUE(Caffeine::cublas_handle());
}

TEST_F(TestCommon, TestVslStream) {
        EXPECT_TRUE(Caffeine::vsl_stream());
    }

TEST_F(TestCommon, TestBresMode){
    EXPECT_EQ(Caffeine::CPU, Caffeine::mode());
    Caffeine::set_mode(Caffeine::GPU);
    EXPECT_EQ(Caffeine::GPU, Caffeine::mode());
}
}// namespace caffeine
