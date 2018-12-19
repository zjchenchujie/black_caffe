//
// Created by chujie on 12/3/18.
//

#ifndef BLACK_CAFFE_VISION_LAYERS_HPP
#define BLACK_CAFFE_VISION_LAYERS_HPP
#include "caffeine/layer.hpp"
#include "caffeine/filler.hpp"

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


//        virtual void Predict_cpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);
//        virtual void Predict_gpu(const std::vector<Blob<Dtype>* >& bottom, std::vector<Blob<Dtype>* >* top);
    }; // relu_layer class

    template <typename Dtype>
    class DropoutLayer : public NeuronLayer<Dtype>{
    public:
        explicit DropoutLayer(LayerParameter& param)
        :NeuronLayer<Dtype>(param) {};
        virtual void SetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);

        virtual Dtype Backward_cpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);

    private:
        shared_ptr<SyncedMemory> rand_vec_;
        float threshold_;
        float scale_;
        unsigned int uint_thres_;
    };

    template <typename  Dtype>
    class InnerProductLayer : public Layer<Dtype>{
    public:
        explicit InnerProductLayer(LayerParameter& param)
        :Layer<Dtype>(param){};
        virtual void SetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);

        virtual Dtype Backward_cpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);

        int M_;
        int K_;
        int N_;
        bool biasterm_;
        shared_ptr<SyncedMemory> bias_multiplier_;
    }; // InnerProductLayer

} //namespace caffeine

#endif //BLACK_CAFFE_VISION_LAYERS_HPP
