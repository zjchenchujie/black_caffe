//
// Created by chujie on 12/2/18.
//

#include "caffeine/base.hpp"

namespace caffeine{
    template <typename Dtype>
    inline void Layer<Dtype>::Forward(const vector<caffeine::Blob<Dtype>* > &bottom, vector<caffeine::Blob<Dtype>* >* top) {
        switch(Caffeine::mode()){
            case Caffeine::CPU:
                Forward_cpu(bottom, top);
                break;
            case Caffeine::GPU:
                Forward_gpu(bottom, top);
                break;
            default:
                CHECK(false);
        }
    };

    template <typename Dtype>
    inline void Layer<Dtype>::Backward(const vector<caffeine::Blob<Dtype> *> &bottom, vector<caffeine::Blob<Dtype> *> *top, bool propagate_down) {
        switch(Caffeine::mode()){
            case Caffeine::CPU:
                Backward_cpu(bottom, top);
                break;
            case Caffeine::GPU:
                Backward_gpu(bottom, top);
                break;
            default:
                CHECK(false);
        }
    };

    template <typename Dtype>
    inline void Layer<Dtype>::Predict(const vector<caffeine::Blob<Dtype> *> &bottom,
                                      vector<caffeine::Blob<Dtype> *> *top) {
        switch(Caffeine::mode()){
            case Caffeine::CPU:
                Predict_cpu(bottom, top);
                break;
            case Caffeine::GPU:
                Predict_gpu(bottom, top);
                break;
            default:
                CHECK(false);
        }
    };


} // namespace caffeine

