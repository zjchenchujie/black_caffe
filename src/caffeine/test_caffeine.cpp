//
// Created by chujie on 11/30/18.
//
#include "gtest/gtest.h"
#include <glog/logging.h>

int main(int argc, char** argv){
    ::testing::InitGoogleTest(&argc, argv);
    ::google::InitGoogleLogging(argv[0]);
    return RUN_ALL_TESTS();
}