//
// Created by chujie on 18-11-23.
//

#ifndef BLACK_CAFFE_BASE_HPP
#define BLACK_CAFFE_BASE_HPP

#include "caffeine/proto/layer_param.pb.h"
#include <vector>
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"

using std::vector;

namespace caffeine {
template <typename Dtype>
class Layer{
public:
    explicit Layer(const LayerParameter& param)
    :layer_param_(param){};
    virtual ~Layer();
    virtual void SetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top) = 0;
    inline void Forward(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
    inline void Predict(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
    inline void Backward(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top, bool propagate_down);
protected:
    // The ptotobuf that stores the layer parameters
    LayerParameter layer_param_;
    // The vector that stores the parameters as a set of blobs
    vector<Blob<Dtype> > blobs;
    virtual void Forward_cpu(vector<const Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top) = 0;
    virtual void Forward_gpu(vector<const Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top){
        LOG(WARNING) << "Using CPU code as backup.";
        Forward_cpu(bottom, top);
    }
    virtual void Backward_cpu(vector<Blob<Dtype>* >& bottom, vector<const Blob<Dtype>* >* top) = 0;
    virtual void Backward_gpu(vector<Blob<Dtype>* >& bottom, vector<const Blob<Dtype>* >* top){
        LOG(WARNING) << "Using CPU code as backup.";
        Backward_cpu(bottom, top);
    }

    virtual void Predict_cpu(vector<const Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top){
        Forward_cpu(bottom, top);
    };
    virtual void Predict_gpu(vector<const Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top) =0;
}; // class Layer
} // namespace caffeine

#endif //BLACK_CAFFE_BASE_HPP
