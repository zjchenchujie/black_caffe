//
// Created by chujie on 18-11-23.
//

#ifndef BLACK_CAFFE_BASE_HPP
#define BLACK_CAFFE_BASE_HPP

#include "caffeine/proto/layer_param.pb.h"
#include <vector>
#include "caffeine/blob.hpp"

using std::vector;

namespace caffeine {
template <typename Dtype>
class Layer{
public:
    explicit Layer(const LayerParameter& param)
    :initialized_(false), layer_param_(param){};
    ~Layer();
    virtual void SetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) = 0;
    virtual void Forward(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) = 0;
    virtual void Predict(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) = 0;
    virtual void Backward(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top, bool propagate_down) = 0;
protected:
    bool initialized_;
    // The ptotobuf that stores the layer parameters
    LayerParameter layer_param_;
    // The vector that stores the parameters as a set of blobs
    vector<Blob<Dtype> > blobs;
}; // class Layer
} // namespace caffeine

#endif //BLACK_CAFFE_BASE_HPP
