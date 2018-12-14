//
// Created by chujie on 18-11-22.
//

#include "caffeine/common.hpp"
#include "caffeine/blob.hpp"
#include "caffeine/syncedmem.hpp"

#include <glog/logging.h>

namespace caffeine{

template <typename Dtype>
Blob<Dtype>::Blob(const Blob<Dtype>& source){
    if(source.count() == 0){
        Blob();
    }else{
        Reshape(source.num(), source.height(), source.width(), source.channels());
        data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
        diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
        memcpy(data_->mutable_cpu_data(), source.cpu_data(), count_ * sizeof(Dtype));
        memcpy(diff_->mutable_cpu_data(), source.cpu_diff(), count_ * sizeof(Dtype));
    }
}
template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int height, const int width,  const int channels){
    num_ = num;
    height_ = height;
    width_ = width;
    channels_ = channels;
    CHECK_GT(num, 0);
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    CHECK_GT(channels, 0);
    count_ = num_* channels_ * height_ * width_;
    data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
}
template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int height, const int width, const int channels){
    Reshape(num, height, width, channels);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
    CHECK(data_);
    return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
    CHECK(data_);
    return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
    CHECK(diff_);
    return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
    CHECK(diff_);
    return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data(){
    CHECK(data_);
    return (Dtype*)data_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data(){
    CHECK(data_);
    return (Dtype*)data_->mutable_gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff(){
    CHECK(diff_);
    return (Dtype*)diff_->mutable_cpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff(){
    CHECK(diff_);
    return (Dtype*)diff_->mutable_gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::Update() {
    //not implemented yet
    LOG(FATAL) << "Update function not implemented yet.";
}

template <typename Dtype>
void Blob<Dtype>::FromPorto(const caffeine::BlobProto &proto) {
    Reshape(proto.num(), proto.height(), proto.width(), proto.channels());
    // copy data
    Dtype* data_vec = mutable_cpu_data();
    for(int i=0; i<count_; i++){
        data_vec[i] = proto.data(i);
    }

    Dtype* diff_vec = mutable_cpu_diff();
    for(int i=0; i<count_; i++){
        diff_vec[i] = proto.diff(i);
    }
}

template <typename Dtype>
void Blob<Dtype>::ToProto(caffeine::BlobProto *proto) {
    proto->set_num(num_);
    proto->set_height(height_);
    proto->set_width(width_);
    proto->set_channels(channels_);
    proto->clear_data();
    proto->clear_diff();

    const Dtype* data_vec = cpu_data();
    for(int i=0; i<count_; i++){
        proto->add_data(data_vec[i]);
    }

    const Dtype* diff_vec = cpu_diff();
    for(int i=0; i<count_; i++){
        proto->add_diff(diff_vec[i]);
    }
}

INSTANTIATE_CLASS(Blob);

} // namespace caffeine

