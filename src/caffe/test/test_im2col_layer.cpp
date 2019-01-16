//
// Created by chujie on 1/11/19.
//
#include <cstring>
#include <cuda_runtime.h>

#include "caffe/vision_layers.hpp"
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe{
    extern cudaDeviceProp caffe_TEST_CUDA_PROP;

    template <typename Dtype>
class Im2colLayerTest : public ::testing::Test{
protected:
    Im2colLayerTest()
    :blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)), blob_top_(new Blob<Dtype>){
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_);
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~Im2colLayerTest(){}
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>* > blob_bottom_vec_;
    vector<Blob<Dtype>* > blob_top_vec_;
};

    typedef ::testing::Types<float, double> Dtypes;
    TYPED_TEST_CASE(Im2colLayerTest, Dtypes);

    TYPED_TEST(Im2colLayerTest, TestSetUp){
        LayerParameter layer_param;
        layer_param.set_kernelsize(3);
        layer_param.set_stride(2);
        Im2colLayer<TypeParam> layer(layer_param);
        layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
        EXPECT_EQ(this->blob_top_->num(), 2);
        EXPECT_EQ(this->blob_top_->channels(), 27);
        EXPECT_EQ(this->blob_top_->height(), 2);
        EXPECT_EQ(this->blob_top_->width(), 2);
    }

    TYPED_TEST(Im2colLayerTest, TestCPU){
        LayerParameter layer_param;
        layer_param.set_kernelsize(3);
        layer_param.set_stride(2);
        Im2colLayer<TypeParam> layer(layer_param);
        caffe::set_mode(caffe::CPU);
        layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
        layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
        // We are lazy and will only check the top left block
        for (int c = 0; c < 27; ++c) {
            EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
                      this->blob_bottom_->data_at(0, (c / 9), (c / 3) % 3, c % 3));
        }
    }


    TYPED_TEST(Im2colLayerTest, TestGPU) {
        LayerParameter layer_param;
        layer_param.set_kernelsize(3);
        layer_param.set_stride(2);
        Im2colLayer<TypeParam> layer(layer_param);
        caffe::set_mode(caffe::GPU);
        layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
        layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
        // We are lazy and will only check the top left block
        for (int c = 0; c < 27; ++c) {
            EXPECT_EQ(this->blob_bottom_->data_at(0, (c / 9), (c / 3) % 3, c % 3),
                      this->blob_top_->data_at(0, c, 0, 0));
        }
    }

/*
    TYPED_TEST(Im2colLayerTest, TestCPUGradient) {
        LayerParameter layer_param;
        layer_param.set_kernelsize(3);
        layer_param.set_stride(2);
        caffe::set_mode(caffe::CPU);
        Im2colLayer<TypeParam> layer(layer_param);
        GradientChecker<TypeParam> checker(1e-2, 1e-2);
        checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
    }
*/
}

