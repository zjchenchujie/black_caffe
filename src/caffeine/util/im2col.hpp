//
// Created by chujie on 1/10/19.
//

#ifndef BLACK_CAFFE_IM2COL_HPP
#define BLACK_CAFFE_IM2COL_HPP

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
        const int height, const int with, const int ksize, const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
        const int heigt, const int width, const int psize, const int stride, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int psize, const int stride, Dtype* data_im);

#endif //BLACK_CAFFE_IM2COL_HPP
