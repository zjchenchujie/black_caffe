//
// Created by chujie on 12/3/18.
//

#ifndef BLACK_CAFFE_VISION_LAYERS_HPP
#define BLACK_CAFFE_VISION_LAYERS_HPP
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include <leveldb/db.h>

namespace caffe{

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

    template<typename Dtype>
    class PaddingLayer : public Layer<Dtype>{
    public:
        explicit PaddingLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {};
        virtual void SetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom);
        unsigned int PAD_;
        int NUM_;
        int CHANNEL_;
        int HEIGHT_IN_;
        int WIDTH_IN_;
        int HEIGHT_OUT_;
        int WIDTH_OUT_;
    };

    template <typename Dtype>
    class LRNLayer : public Layer<Dtype>{
    public:
        explicit LRNLayer(const LayerParameter& param)
        :Layer<Dtype>(param){}
        virtual void SetUp(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            vector<Blob<Dtype>*>* top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
            const bool propagate_down, vector<Blob<Dtype>*>* bottom);

        // scale_ stores the intermediate summing results
        Blob<Dtype> scale_;
        int size_;
        int pre_pad_;
        Dtype alpha_;
        Dtype beta_;
        int num_;
        int channels_;
        int height_;
        int width_;
    };

    template <typename Dtype>
    class Im2colLayer : public Layer<Dtype> {
    public:
        explicit Im2colLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {};
        virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                           vector<Blob<Dtype>*>* top);
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        int KSIZE_;
        int STRIDE_;
        int CHANNELS_;
        int HEIGHT_;
        int WIDTH_;
    };

    template <typename Dtype>
    class PoolingLayer : public Layer<Dtype>{
    public:
        explicit PoolingLayer(const LayerParameter &param)
        :Layer<Dtype>(param){};
        virtual void SetUp(const vector<Blob<Dtype>* > &bottom, vector<Blob<Dtype>* > *top);
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            vector<Blob<Dtype>*>* top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
            const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        int KSIZE_;
        int STRIDE_;
        int CHANNELS_;
        int HEIGHT_;
        int WIDTH_;
        int POOLED_HEIGHT_;
        int POOLED_WIDTH_;
    };

    template <typename Dtype>
    class ConvolutionLayer : public Layer<Dtype> {
    public:
        explicit ConvolutionLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {};
        virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                           vector<Blob<Dtype>*>* top);
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            vector<Blob<Dtype>*>* top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
            const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        Blob<Dtype> col_bob_;

        int KSIZE_;
        int STRIDE_;
        int NUM_;
        int CHANNELS_;
        int HEIGHT_;
        int WIDTH_;
        int NUM_OUTPUT_;
        int GROUP_;
        Blob<Dtype> col_buffer_;
        shared_ptr<SyncedMemory> bias_multiplier_;
        bool biasterm_;
        int M_;
        int K_;
        int N_;
    };

    template <typename Dtype>
    class DataLayer : public Layer<Dtype>{
    public:
        explicit DataLayer<Dtype>(const LayerParameter &param)
                :Layer<Dtype>(param){};
        virtual void SetUp(const vector<Blob<Dtype>* > &bottom, vector<Blob<Dtype>* > *top);

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>* > &bottom, vector<Blob<Dtype>* > *top);
        virtual void Forward_gpu(const vector<Blob<Dtype>* > &bottom, vector<Blob<Dtype>* > *top);
        virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);
        virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down, vector<Blob<Dtype>*>* bottom);

        shared_ptr<leveldb::DB> db_;
        shared_ptr<leveldb::Iterator> iter_;
        int datum_size_;
    };



} //namespace caffe

#endif //BLACK_CAFFE_VISION_LAYERS_HPP
