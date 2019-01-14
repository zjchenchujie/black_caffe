#include "gtest/gtest.h"
#include "caffeine/syncedmem.hpp"
#include "caffeine/common.hpp"

#include <cuda_runtime.h>

#include "caffeine/test/test_caffeine_main.hpp"


namespace caffeine {

class SyncedMemoryTest: public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization){
    SyncedMemory mem(10);
    EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
    EXPECT_EQ(10, mem.size());
    SyncedMemory *p_mem = new SyncedMemory(10 * sizeof(float));
    EXPECT_EQ(10 * sizeof(float), p_mem->size());
}

TEST_F(SyncedMemoryTest, TestAllocation){
    SyncedMemory mem(10);
    EXPECT_TRUE(mem.cpu_data());
    EXPECT_TRUE(mem.gpu_data());
    EXPECT_TRUE(mem.mutable_gpu_data());
    EXPECT_TRUE(mem.mutable_cpu_data());
}

TEST_F(SyncedMemoryTest, TestCPUWrite){
    SyncedMemory mem(10);
    void *cpu_data = mem.mutable_cpu_data();
    EXPECT_EQ(SyncedMemory::HEAD_AT_CPU, mem.head());
    memset(cpu_data, 1, mem.size());
    for(int i=0; i<mem.size(); i++)
    {
        EXPECT_EQ(1, ((char*)cpu_data)[i]);
    }
    const void *gpu_data = mem.gpu_data();
    EXPECT_EQ(SyncedMemory::SYNCED, mem.head());

    char* recovered_data = new char[10];
    memset((void*)recovered_data, 0, 10);
    CUDA_CHECK(cudaMemcpy((void*)recovered_data, gpu_data, 10, cudaMemcpyDeviceToHost));
    for(int i=0; i<mem.size(); i++){
        EXPECT_EQ(1, ((char*)recovered_data)[i]);
    }

    // do another round
    cpu_data = mem.mutable_cpu_data();
    EXPECT_EQ(SyncedMemory::HEAD_AT_CPU, mem.head());
    memset(cpu_data, 2, mem.size());
    for(int i = 0; i < mem.size(); i++){
        EXPECT_EQ(2, ((char*)cpu_data)[i]);
    }
    gpu_data = mem.mutable_gpu_data();
    EXPECT_EQ(SyncedMemory::HEAD_AT_GPU, mem.head());
    cudaMemcpy((void*)recovered_data, gpu_data, 10, cudaMemcpyDeviceToHost);
    for (int i = 0; i < mem.size(); ++i) {
        EXPECT_EQ(((char*)recovered_data)[i], 2);
    }

    delete[] recovered_data;
}

TEST_F(SyncedMemoryTest, TestGPUWrite){
    SyncedMemory mem(10);
    void *gpu_data = mem.mutable_gpu_data();
    EXPECT_EQ(SyncedMemory::HEAD_AT_GPU, mem.head());
    CUDA_CHECK(cudaMemset(gpu_data, 1, mem.size()));
    const void* cpu_data = mem.cpu_data();
    for(int i=0; i<mem.size(); i++){
        EXPECT_EQ(((char*)cpu_data)[i], 1);
    }
    EXPECT_EQ(SyncedMemory::SYNCED, mem.head());

    gpu_data = mem.mutable_gpu_data();
    EXPECT_EQ(SyncedMemory::HEAD_AT_GPU, mem.head());
    CUDA_CHECK(cudaMemset(gpu_data, 2, mem.size()));
    cpu_data = mem.cpu_data();
    for(int i = 0; i < mem.size(); i++){
        EXPECT_EQ(2, ((char*)cpu_data)[i]);
    }
    EXPECT_EQ(SyncedMemory::SYNCED, mem.head());
}

}

