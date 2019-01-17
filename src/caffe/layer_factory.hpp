//
// Created by chujie on 1/17/19.
//

#ifndef BLACK_CAFFE_LAYER_FACTORY_HPP
#define BLACK_CAFFE_LAYER_FACTORY_HPP

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "proto/layer_param.pb.h"

#include <string>

namespace caffe {

    template <typename Dtype>
    Layer<Dtype>* GetLayer(const LayerParameter &param){
        const std::string &type = param.type();
        if(type == "conv"){
            return new ConvolutionLayer<Dtype>(param);
        }else if(type == "dropout"){
            return new DropoutLayer<Dtype>(param);
        }else if(type == "im2col"){
            return new Im2colLayer<Dtype>(param);
        }else if(type == "innerproduct"){
            return new InnerProductLayer<Dtype>(param);
        }else if(type == "lrn"){
            return new LRNLayer<Dtype>(param);
        }else if(type == "padding"){
            return new PaddingLayer<Dtype>(param);
        }else if(type == "pool"){
            return new PoolingLayer<Dtype>(param);
        }else if(type == "relu"){
            return new ReLULayer(param);
        }else {
            LOG(FATAL) << "Unkonwn layer name: " << type;
        }
        return (Layer<Dtype> *)(NULL);
    }
} // namespace caffe


#endif //BLACK_CAFFE_LAYER_FACTORY_HPP
