#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/util/math_functions.hpp"

namespace caffeine {

    template <typename Dtype>
    __global__ void LRNFillScale(const int nthreads, const Dtype* in,
                                 const int num, const int channels, const int height,
                                 const int width, const int size, const Dtype alpha_over_size,
                                 Dtype* scale) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < nthreads) {
            // find out the local offset
            int w = index % width;
            int h = (index / width) % height;
            int n = index / width / height;
            int offset = (n * channels * height + h) * width + w;
            int step = height * width;
            in += offset;
            scale += offset;
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            Dtype accum_scale = 0;
            // fill the scale at [n, :, h, w]
            // accumulate values
            while (head < post_pad) {
                accum_scale += in[head * step] * in[head * step];
                ++head;
            }
            // until we reach size, nothing needs to be subtracted
            while (head < size) {
                accum_scale += in[head * step] * in[head * step];
                scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
                ++head;
            }
            // both add and subtract
            while (head < channels) {
                accum_scale += in[head * step] * in[head * step];
                accum_scale -= in[(head - size) * step] * in[(head - size) * step];
                scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
                ++head;
            }
            // subtract only
            while (head < channels + post_pad) {
                accum_scale -= in[(head - size) * step] * in[(head - size) * step];
                scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
                ++head;
            }
        }
    }


// TODO: check if it would be faster to just put it into the previous kernel.
    template <typename Dtype>
    __global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
                                     const Dtype* scale, const Dtype negative_beta, Dtype* out) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < nthreads) {
            out[index] = in[index] * pow(scale[index], negative_beta);
        }
    }

    template <typename Dtype>
    void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top) {
        // First, compute scale
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = (*top)[0]->mutable_gpu_data();
        Dtype* scale_data = scale_.mutable_gpu_data();
        // We will launch one kernel for each pixel location, and have the kernel
        // go through all the channels.
        int n_threads = num_ * height_ * width_;
        LRNFillScale<<<CAFFEINE_GET_BLOCKS(n_threads), CAFFEINE_CUDA_NUM_THREADS>>>(
                n_threads, bottom_data, num_, channels_, height_, width_, size_,
                        alpha_ / size_, scale_data);
        CUDA_POST_KERNEL_CHECK;
        n_threads = bottom[0]->count();
        LRNComputeOutput<<<CAFFEINE_GET_BLOCKS(n_threads), CAFFEINE_CUDA_NUM_THREADS>>>(
                n_threads, bottom_data, scale_data, -beta_, top_data);
        CUDA_POST_KERNEL_CHECK;
    }


    template <typename Dtype>
    __global__ void LRNComputeDiff(const int nthreads, const Dtype* bottom_data,
                                   const Dtype* top_data, const Dtype* scale, const Dtype* top_diff,
                                   const int num, const int channels, const int height,
                                   const int width, const int size, const Dtype negative_beta,
                                   const Dtype cache_ratio,
                                   Dtype* bottom_diff) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < nthreads) {
            // find out the local offset
            int w = index % width;
            int h = (index / width) % height;
            int n = index / width / height;
            int offset = (n * channels * height + h) * width + w;
            int step = height * width;
            bottom_data += offset;
            top_data += offset;
            scale += offset;
            top_diff += offset;
            bottom_diff += offset;
            int head = 0;
            int pre_pad = size - (size + 1) / 2;
            int post_pad = size - pre_pad - 1;
            Dtype accum_ratio = 0;
            // accumulate values
            while (head < post_pad) {
                accum_ratio += top_diff[head * step] * top_data[head * step] /
                               scale[head * step];
                ++head;
            }
            // until we reach size, nothing needs to be subtracted
            while (head < size) {
                accum_ratio += top_diff[head * step] * top_data[head * step] /
                               scale[head * step];
                bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
                                                        * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
                                                                                                                bottom_data[(head - post_pad) * step] * accum_ratio;
                ++head;
            }
            // both add and subtract
            while (head < channels) {
                accum_ratio += top_diff[head * step] * top_data[head * step] /
                               scale[head * step];
                accum_ratio -= top_diff[(head - size) * step] *
                               top_data[(head - size) * step] / scale[(head - size) * step];
                bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
                                                        * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
                                                                                                                bottom_data[(head - post_pad) * step] * accum_ratio;
                ++head;
            }
            // subtract only
            while (head < channels + post_pad) {
                accum_ratio -= top_diff[(head - size) * step] *
                               top_data[(head - size) * step] / scale[(head - size) * step];
                bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
                                                        * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
                                                                                                                bottom_data[(head - post_pad) * step] * accum_ratio;
                ++head;
            }
        }
    }

    template <typename Dtype>
    Dtype LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
        int n_threads = num_ * height_ * width_;
        LRNComputeDiff<<<CAFFEINE_GET_BLOCKS(n_threads), CAFFEINE_CUDA_NUM_THREADS>>>(
                n_threads, (*bottom)[0]->gpu_data(), top[0]->gpu_data(),
                        scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
                        size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
                        (*bottom)[0]->mutable_gpu_diff());
        return Dtype(0.);
    }


    INSTANTIATE_CLASS(LRNLayer);

}  // namespace caffeine
