//
// Created by chujie on 1/11/19.
//

#ifndef BLACK_CAFFE_TEST_CAFFEINE_MAIN_HPP
#define BLACK_CAFFE_TEST_CAFFEINE_MAIN_HPP

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace caffeine {

    cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;

}  // namespace caffeine

using namespace caffeine;
using namespace std;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::InitGoogleLogging(argv[0]);
    // Before starting testing, let's first print out a few cuda defice info.
    int device;
    cudaGetDeviceCount(&device);
    cout << "Cuda number of devices: " << device << endl;
    if (argc > 1) {
        // Use the given device
        device = atoi(argv[1]);
        cudaSetDevice(device);
        cout << "Setting to use device " << device << endl;
    }
    cudaGetDevice(&device);
    cout << "Current device id: " << device << endl;
    cudaGetDeviceProperties(&CAFFEINE_TEST_CUDA_PROP, device);
    // invoke the test.
    return RUN_ALL_TESTS();
}

#endif //BLACK_CAFFE_TEST_CAFFEINE_MAIN_HPP
