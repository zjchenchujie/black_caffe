//
// Created by chujie on 18-11-22.
//


#ifndef BLACK_CAFFE_BLOB_HPP
#define BLACK_CAFFE_BLOB_HPP

#include <memory>
#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"
#include "caffeine/proto/layer_param.pb.h"

namespace caffeine {

template <typename Dtype>
class Blob{
public:
    Blob()
        :num_(0), channels_(0), height_(0), width_(0), count_(0), data_(), diff_() {};
    explicit Blob(const int num, const int channels, const int height, const int width);
    Blob(const Blob<Dtype>& source);

    ~Blob(){
//        delete data_;
//        delete diff_;
    };
    void Reshape(const int num, const int channels, const int height, const int width);
    inline int num() const { return num_; }
    inline int channels() const { return channels_; }
    inline int height() const { return height_; }
    inline int width() const { return width_; }
    inline int count() const { return count_; }

    const Dtype* cpu_data() const;
    const Dtype* gpu_data() const;
    const Dtype* cpu_diff() const;
    const Dtype* gpu_diff() const;
    Dtype* mutable_cpu_data();
    Dtype* mutable_gpu_data();
    Dtype* mutable_cpu_diff();
    Dtype* mutable_gpu_diff();

    void Update();
    void FromPorto(const BlobProto& proto);
    void ToProto(BlobProto* proto);

private:
    shared_ptr<SyncedMemory> data_;
    shared_ptr<SyncedMemory> diff_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;

}; // class Blob

} // namespace caffeine

#endif //BLACK_CAFFE_BLOB_HPP
