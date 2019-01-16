//
// Created by chujie on 12/3/18.
// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.
//

#ifndef BLACK_CAFFE_FILLER_HPP
#define BLACK_CAFFE_FILLER_HPP

#include <mkl.h>
#include "caffe/common.hpp"
#include "caffe/proto/layer_param.pb.h"
#include "caffe/blob.hpp"

namespace caffe{

template <typename Dtype>
class Filler {
public:
    Filler(const FillerParameter& param)
    :filler_param_(param){};
    ~Filler(){};
    virtual void Fill(Blob<Dtype>* blob) = 0;
protected:
    FillerParameter filler_param_;
};

template <typename Dtype>
class ConstantFiller : public Filler<Dtype>{
public:
    ConstantFiller(const FillerParameter& param)
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
    UniformFiller(const FillerParameter& param)
    :Filler<Dtype>(param){};
    virtual void Fill(Blob<Dtype>* blob){
        Dtype* data = blob->mutable_cpu_data();
        const int count = blob->count();
        const Dtype value = this->filler_param_.value();
        CHECK(count);
        switch(sizeof(Dtype)) {
            case sizeof(float):
                VSL_CHECK(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, caffe::vsl_stream(),
                                       count, (float*)data, this->filler_param_.min(),
                                       this->filler_param_.max()));
                break;
            case sizeof(double):
                VSL_CHECK(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, caffe::vsl_stream(),
                                       count, (double*)data, this->filler_param_.min(), this->filler_param_.max()));
                break;
            default:
                CHECK(false) << "Unknown dtype. ";
        }
    }
}; // class UniformFiller

template <typename Dtype>
class GaussianFiller : public Filler<Dtype>{
public:
    GaussianFiller(const FillerParameter& param)
    :Filler<Dtype>(param){};
    virtual void Fill(Blob<Dtype>* blob){
        Dtype* data = blob->mutable_cpu_data();
        const int count = blob->count();
        const Dtype value = this->filler_param_.value();
        CHECK(count);
        switch(sizeof(Dtype)){
            case sizeof(float):
                VSL_CHECK(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
                                        caffe::vsl_stream(), count, (float*)data,
                                        this->filler_param_.mean(), this->filler_param_.std()));
                break;
            case sizeof(double):
                VSL_CHECK(vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
                                        caffe::vsl_stream(), count, (double*)data,
                                        this->filler_param_.mean(), this->filler_param_.std()));
                break;
            default:
                CHECK(false) << "Unknown dtype.";
        } //switch
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
