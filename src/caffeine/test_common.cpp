//
// Created by chujie on 12/2/18.
//
#include "caffeine/common.hpp"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cstring>

namespace caffeine{
class TestCommon : public ::testing::Test{};

TEST_F(TestCommon, TestCublasHandler){
    EXPECT_TRUE(Caffeine::cublas_handle());
}

TEST_F(TestCommon, TestBresMode){
    EXPECT_EQ(Caffeine::CPU, Caffeine::mode());
    Caffeine::set_mode(Caffeine::GPU);
    EXPECT_EQ(Caffeine::GPU, Caffeine::mode());
}
}// namespace caffeine
