//
// Created by chujie on 1/11/19.
//

#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/util/math_functions.hpp"


namespace caffeine {

    template <typename Dtype>
    void LRNLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                vector<Blob<Dtype>*>* top) {
        CHECK_EQ(bottom.size(), 1) <<
                                   "Local Response Normalization Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) <<
                                 "Local Response Normalization Layer takes a single blob as output.";
        num_ = bottom[0]->num();
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        (*top)[0]->Reshape(num_, channels_, height_, width_);
        scale_.Reshape(num_, channels_, height_, width_);
        size_ = this->layer_param_.local_size();
        pre_pad_ = (size_ - 1) / 2;
        alpha_ = this->layer_param_.alpha();
        beta_ = this->layer_param_.beta();
    };

    template <typename Dtype>
    void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = (*top)[0]->mutable_cpu_data();
        Dtype* scale_data = scale_.mutable_cpu_data();
        // start with the constant value
        for (int i = 0; i < scale_.count(); ++i) {
            scale_data[i] = 1.;
        }
        Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
        Dtype* padded_square_data = padded_square.mutable_cpu_data();
        memset(padded_square_data, 0, sizeof(Dtype) * padded_square.count());
        Dtype alpha_over_size = alpha_ / size_;
        // go through the images
        for (int n = 0; n < num_; ++n) {
            // compute the padded square
            caffeine_sqr(channels_ * height_ * width_,
                         bottom_data + bottom[0]->offset(n),
                         padded_square_data + padded_square.offset(0, pre_pad_));
            // Create the first channel scale
            for (int c = 0; c < size_; ++c) {
                caffeine_axpy<Dtype>(height_ * width_, alpha_over_size,
                                     padded_square_data + padded_square.offset(0, c),
                                     scale_data + scale_.offset(n, 0));
            }
            for (int c = 1; c < channels_; ++c) {
                // copy previous scale
                caffeine_copy<Dtype>(height_ * width_,
                                     scale_data + scale_.offset(n, c - 1),
                                     scale_data + scale_.offset(n, c));
                // add head
                caffeine_axpy<Dtype>(height_ * width_, alpha_over_size,
                                     padded_square_data + padded_square.offset(0, c + size_ - 1),
                                     scale_data + scale_.offset(n, c));
                // subtract tail
                caffeine_axpy<Dtype>(height_ * width_, -alpha_over_size,
                                     padded_square_data + padded_square.offset(0, c - 1),
                                     scale_data + scale_.offset(n, c));
            }
        }

        // In the end, compute output
        caffeine_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
        caffeine_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
    }

    template <typename Dtype>
    Dtype LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* bottom_data = (*bottom)[0]->cpu_data();
        const Dtype* scale_data = scale_.cpu_data();
        Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
        Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
        Blob<Dtype> accum_ratio(1, 1, height_, width_);
        Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
        Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
        // We hack a little bit by using the diff() to store an additional result
        Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
        memset(padded_ratio_data, 0, sizeof(Dtype) * padded_ratio.count());
        Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

        caffeine_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
        caffeine_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

        // go through individual data
        int inverse_pre_pad = size_ - (size_ + 1) / 2;
        for (int n = 0; n < num_; ++n) {
            // first, compute diff_i * y_i / s_i
            caffeine_mul<Dtype>(scale_.count(), top_diff, top_data,
                                padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
            caffeine_div<Dtype>(scale_.count(),
                                padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
                                scale_data,
                                padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
            // Now, compute the accumulated ratios and the bottom diff
            memset(accum_ratio_data, 0, sizeof(Dtype) * accum_ratio.count());
            for (int c = 0; c < size_ - 1; ++c) {
                caffeine_axpy<Dtype>(height_ * width_, 1.,
                                     padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
            }
            for (int c = 0; c < channels_; ++c) {
                caffeine_axpy<Dtype>(height_ * width_, 1.,
                                     padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
                                     accum_ratio_data);
                // compute bottom diff
                caffeine_mul<Dtype>(height_ * width_,
                                    bottom_data + top[0]->offset(n, c),
                                    accum_ratio_data, accum_ratio_times_bottom);
                caffeine_axpy<Dtype>(height_ * width_, -cache_ratio_value,
                                     accum_ratio_times_bottom, bottom_diff + top[0]->offset(n,c));
                caffeine_axpy<Dtype>(height_ * width_, -1.,
                                     padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
            }
        }
        return Dtype(0.);
    }

    INSTANTIATE_CLASS(LRNLayer);


}  // namespace caffeine