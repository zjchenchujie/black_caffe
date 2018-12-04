//
// Created by chujie on 12/3/18.
//

#ifndef BLACK_CAFFE_VISION_LAYERS_HPP
#define BLACK_CAFFE_VISION_LAYERS_HPP
#include "caffeine/layer.hpp"

namespace caffeine{

    template <typename Dtype>
    class NeuronLayer : public Layer<Dtype>{
    public:
        explicit NeuronLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {};
        virtual void SetUp(const std::vector<Blob<Dtype>*>& bottom, std::vector<Blob<Dtype>*>* top);
    };

    template <typename Dtype>
    class ReLULayer : public NeuronLayer<Dtype>{
    public:
        explicit ReLULayer(const LayerParameter& param)
                : NeuronLayer<Dtype>(param) {};
    protected:
        virtual void Forward_cpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);
        virtual void Forward_gpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);

        virtual Dtype Backward_cpu(const std::vector<Blob<Dtype>*>& top, const bool propagate_down, std::vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const std::vector<Blob<Dtype>*>& top, const bool propagate_down, std::vector<Blob<Dtype>*>* bottom);


        virtual void Predict_cpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);
        virtual void Predict_gpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);
    };
} //namespace caffeine

#endif //BLACK_CAFFE_VISION_LAYERS_HPP
