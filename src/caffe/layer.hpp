#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/layer_param.pb.h"

using std::vector;

namespace caffe {

    template <typename Dtype>
    class Layer {
    public:
        // You should not implement your own constructor. Any set up code should go
        // to SetUp(), where the dimensions of the bottom blobs are provided to the
        // layer.
        explicit Layer(const LayerParameter& param)
                : layer_param_(param) {};
        virtual ~Layer(){};
        // SetUp: your function should implement this.
        virtual void SetUp(const std::vector<Blob<Dtype>*>& bottom,
                           std::vector<Blob<Dtype>*>* top) = 0;

        // Forward, backward wrappers. You should implement the cpu and
        // gpu specific implementations instead, and should not change these
        // functions.
        inline void Forward(const vector<Blob<Dtype>*>& bottom,
                            vector<Blob<Dtype>*>* top);
        inline Dtype Backward(const std::vector<Blob<Dtype>*>& top,
                              const bool propagate_down,
                              std::vector<Blob<Dtype>*>* bottom);
        vector<Blob<Dtype> >& params(){ return blobs_; };

    protected:
        // The protobuf that stores the layer parameters
        LayerParameter layer_param_;
        // The vector that stores the parameters as a set of blobs.
        vector<Blob<Dtype> > blobs_;

        // Forward functions
        virtual void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                                 std::vector<Blob<Dtype>*>* top) = 0;
        // If no gpu code is provided, we will simply use cpu code.
        virtual void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                                 std::vector<Blob<Dtype>*>* top) {
            LOG(WARNING) << "Using CPU code as backup.";
            Forward_cpu(bottom, top);
        };

        // Backward functions: the backward function will compute the gradients for
        // any parameters and also for the bottom blobs if propagate_down is true.
        // It will return the loss produced from this layer.
        virtual Dtype Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                                   const bool propagate_down,
                                   std::vector<Blob<Dtype>*>* bottom) = 0;
        virtual Dtype Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                                   const bool propagate_down,
                                   std::vector<Blob<Dtype>*>* bottom) {
            LOG(WARNING) << "Using CPU code as backup.";
            return Backward_cpu(top, propagate_down, bottom);
        };

    };  // class Layer

    // Forward, backward and predict wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
    template <typename Dtype>
    inline void Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top) {
        switch(Caffe::mode()) {
            case Caffe::CPU:
                Forward_cpu(bottom, top);
                break;
            case Caffe::GPU:
                Forward_gpu(bottom, top);
                break;
            default:
                LOG(FATAL) << "Unknown Caffe mode.";
        }
    };

    template <typename Dtype>
    inline Dtype Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
                                        const bool propagate_down,
                                        vector<Blob<Dtype>*>* bottom) {
        switch(Caffe::mode()) {
            case Caffe::CPU:
                return Backward_cpu(top, propagate_down, bottom);
            case Caffe::GPU:
                return Backward_gpu(top, propagate_down, bottom);
            default:
                LOG(FATAL) << "Unknown Caffe mode.";
        }
    };

}  // namespace caffe

#endif  // Caffe_LAYER_H_
