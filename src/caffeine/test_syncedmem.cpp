#include "gtest/gtest.h"
#include "caffeine/syncedmem.hpp"
#include "caffeine/common.hpp"

#include <cuda_runtime.h>


namespace caffeine {

class SyncedMemoryTest: public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization){
    SyncedMemory mem(10);
    EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
}

TEST_F(SyncedMemoryTest, TestAllocation){
    SyncedMemory mem(10);
    EXPECT_NE(mem.cpu_data(), (void*)NULL);
    EXPECT_NE(mem.gpu_data(), (void*)NULL);
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
}

}

int main(int argc, char** argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
