
#ifndef BLACK_CAFFE_SYNCEDMEN_HPP
#define BLACK_CAFFE_SYNCEDMEN_HPP

#include <cstring>

namespace caffe {

class SyncedMemory {
public:
    SyncedMemory()
        :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED){};
    SyncedMemory(size_t size)
        :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED){};
    ~SyncedMemory();
    const void* cpu_data();
    const void* gpu_data();
    void* mutable_cpu_data();
    void* mutable_gpu_data();
    enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
    SyncedHead head() { return head_;}
    size_t size(){ return size_; }
private:
    void to_cpu();
    void to_gpu();
    void *cpu_ptr_;
    void *gpu_ptr_;
    size_t size_;
    SyncedHead head_;
};    // class SyncedMemory
} // namespace caffe

#endif //BLACK_CAFFE_SYNCEDMEN_HPP
