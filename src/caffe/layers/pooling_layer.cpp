//
// Created by chujie on 1/17/19.
//
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include <cfloat>
#include <algorithm>

using std::min;
using std::max;

namespace caffe{

    const float CAFFE_MAX_POOLING_THRESHOLD = 1e-8;

    template <typename Dtype>
    void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype> *> &bottom, vector<Blob<Dtype> *> *top) {
        CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) << "PoolingLayer takes a single blob as output.";
        KSIZE_ = this->layer_param_.kernelsize();
        STRIDE_ = this->layer_param_.stride();
        CHANNELS_ = bottom[0]->channels();
        HEIGHT_ = bottom[0]->height();
        WIDTH_ = bottom[0]->width();
        POOLED_HEIGHT_ = int(ceil(float(HEIGHT_ - KSIZE_) / STRIDE_)) + 1;
        POOLED_WIDTH_ = int(ceil(float(WIDTH_ - KSIZE_) / STRIDE_)) + 1;
        (*top)[0]->Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
    }

    template <typename Dtype>
    void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          vector<Blob<Dtype> *> *top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = (*top)[0]->mutable_cpu_data();
        const int top_count = (*top)[0]->count();
        switch (this->layer_param_.pool()){
            case LayerParameter_PoolMethod_MAX:
                for (int i = 0; i < top_count; ++i) {
                    top_data[i] = -FLT_MAX;
                }
                for(int n = 0; n < bottom[0]->num(); ++n){
                    for(int c = 0; c < CHANNELS_; ++c){
                        for(int ph = 0; ph < POOLED_HEIGHT_; ++ph){
                            for(int pw = 0; pw < POOLED_WIDTH_; ++pw){
                                int hstart = ph * STRIDE_;
                                int wstart = pw * STRIDE_;
                                int hend = min(hstart + KSIZE_, HEIGHT_);
                                int wend = min(wstart + KSIZE_, WIDTH_);
                                for(int h = hstart; h < hend; ++h){
                                    for(int w = wstart; w < wend; ++w){
                                        top_data[ph * POOLED_WIDTH_ + pw] =
                                                std::max(top_data[ph * POOLED_WIDTH_ + pw],
                                                        bottom_data[h * WIDTH_ + w]);
                                    }
                                }
                            }
                        }
                        top_data += (*top)[0]->offset(0, 1);
                        bottom_data += bottom[0]->offset(0, 1);
                    }
                }

                break;

            case LayerParameter_PoolMethod_AVE:
                for(int i = 0; i < top_count; ++i){
                    top_data[i] = 0;
                }
                //main loop
                for(int n = 0; n < bottom[0]->num(); ++n){
                    for(int c = 0; c < CHANNELS_; ++c){
                        for(int ph = 0; ph < POOLED_HEIGHT_; ++ph){
                            for(int pw = 0; pw < POOLED_WIDTH_; ++pw){
                                int hstart = ph * STRIDE_;
                                int wstart = pw * STRIDE_;
                                int hend = min(hstart + KSIZE_, HEIGHT_);
                                int wend = min(wstart + KSIZE_, WIDTH_);
                                for(int h = hstart; h < hend; ++h){
                                    for(int w = wstart; w < wend; ++w){
                                        top_data[ph * POOLED_WIDTH_ + pw] += bottom_data[h * WIDTH_ + w];
                                    }
                                }
                            }
                        }
                        // compute offset
                        bottom_data += bottom[0]->offset(0, 1);
                        top_data += (*top)[0]->offset(0, 1);
                    }
                }
                // Our implementation simply divides the pooled values by KSIZE^2,
                // regardless of the actual pooling region. This would allow one to not
                // trust too much on the border pooling regions, but I am not sure what
                // benefit / harm it would bring to the actual code.
                caffe_scal<Dtype>(top_count, Dtype(1.) / KSIZE_ / KSIZE_,
                                  (*top)[0]->mutable_cpu_data());
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
    }

    template <typename Dtype>
    Dtype PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
        if (!propagate_down) {
            return Dtype(0.);
        }
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* bottom_data = (*bottom)[0]->cpu_data();
        Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
        // Different pooling methods. We explicitly do the switch outside the for
        // loop to save time, although this results in more codes.
        memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
        switch (this->layer_param_.pool()) {
            case LayerParameter_PoolMethod_MAX:
                // The main loop
                for (int n = 0; n < top[0]->num(); ++n) {
                    for (int c = 0; c < CHANNELS_; ++c) {
                        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
                            for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
                                int hstart = ph * STRIDE_;
                                int wstart = pw * STRIDE_;
                                int hend = min(hstart + KSIZE_, HEIGHT_);
                                int wend = min(wstart + KSIZE_, WIDTH_);
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        bottom_diff[h * WIDTH_ + w] +=
                                                top_diff[ph * POOLED_WIDTH_ + pw] *
                                                (bottom_data[h * WIDTH_ + w] >=
                                                 top_data[ph * POOLED_WIDTH_ + pw] -
                                                 CAFFE_MAX_POOLING_THRESHOLD);
                                    }
                                }
                            }
                        }
                        // offset
                        bottom_data += (*bottom)[0]->offset(0, 1);
                        top_data += top[0]->offset(0, 1);
                        bottom_diff += (*bottom)[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                    }
                }
                break;
            case LayerParameter_PoolMethod_AVE:
                // The main loop
                for (int n = 0; n < top[0]->num(); ++n) {
                    for (int c = 0; c < CHANNELS_; ++c) {
                        for (int ph = 0; ph < POOLED_HEIGHT_; ++ph) {
                            for (int pw = 0; pw < POOLED_WIDTH_; ++pw) {
                                int hstart = ph * STRIDE_;
                                int wstart = pw * STRIDE_;
                                int hend = min(hstart + KSIZE_, HEIGHT_);
                                int wend = min(wstart + KSIZE_, WIDTH_);
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        bottom_diff[h * WIDTH_ + w] +=
                                                top_diff[ph * POOLED_WIDTH_ + pw];
                                    }
                                }
                            }
                        }
                        // offset
                        bottom_data += (*bottom)[0]->offset(0, 1);
                        top_data += top[0]->offset(0, 1);
                        bottom_diff += (*bottom)[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                    }
                }
                // Our implementation simply divides the pooled values by KSIZE^2,
                // regardless of the actual pooling region. This would allow one to not
                // trust too much on the border pooling regions, but I am not sure what
                // benefit / harm it would bring to the actual code.
                caffe_scal<Dtype>((*bottom)[0]->count(), Dtype(1.) / KSIZE_ / KSIZE_,
                                  (*bottom)[0]->mutable_cpu_diff());
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
        return Dtype(0.);
    }


    INSTANTIATE_CLASS(PoolingLayer);



} // namespace caffe
