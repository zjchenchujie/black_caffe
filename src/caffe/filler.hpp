//
// Created by chujie on 12/3/18.
// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.
//

#ifndef BLACK_CAFFE_FILLER_HPP
#define BLACK_CAFFE_FILLER_HPP

#include <mkl.h>
#include <string>
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/layer_param.pb.h"
#include "caffe/blob.hpp"

namespace caffe{

template <typename Dtype>
class Filler {
public:
    explicit Filler(const FillerParameter& param)
    :filler_param_(param){};
    virtual ~Filler(){};
    virtual void Fill(Blob<Dtype>* blob) = 0;
protected:
    FillerParameter filler_param_;
};

template <typename Dtype>
class ConstantFiller : public Filler<Dtype>{
public:
    explicit ConstantFiller(const FillerParameter& param)
    :Filler<Dtype>(param){};
    virtual void Fill(Blob<Dtype>* blob){
        Dtype* data = blob->mutable_cpu_data();
        const int count = blob->count();
        const Dtype value = this->filler_param_.value();
        CHECK(count);
        for(int i=0; i<count; i++){
            data[i] = value;
        }
    }
};

template <typename Dtype>
class UniformFiller : public Filler<Dtype>{
public:
    explicit UniformFiller(const FillerParameter& param)
    :Filler<Dtype>(param){};
    virtual void Fill(Blob<Dtype>* blob){
        DCHECK(blob->count());
        caffe_vRngUniform<Dtype>(blob->count(), blob->mutable_cpu_data(),
                Dtype(this->filler_param_.min()),
                Dtype(this->filler_param_.max()));
    }
}; // class UniformFiller

template <typename Dtype>
class GaussianFiller : public Filler<Dtype>{
public:
    explicit GaussianFiller(const FillerParameter& param)
    :Filler<Dtype>(param){};
    virtual void Fill(Blob<Dtype>* blob){
        CHECK(blob->count());
        caffe_vRngGaussian<Dtype>(blob->count(), blob->mutable_cpu_data(),
                Dtype(this->filler_param_.mean()),
                Dtype(this->filler_param_.std()));
    } // func Fill
}; // class GaussianFiller

template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param){
    const std::string& type = param.type();
    if(type == "constant"){
        return new ConstantFiller<Dtype>(param);
    }else if(type == "uniform"){
        return new UniformFiller<Dtype>(param);
    }else if(type == "gaussian"){
        return new GaussianFiller<Dtype>(param);
    } else{
        CHECK(false) << "Unknown filler name: " << param.type();
    }
    return (Filler<Dtype>*)(NULL);
}

} // namespace caffe

#endif //BLACK_CAFFE_FILLER_HPP
