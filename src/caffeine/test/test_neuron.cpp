//
// Created by chujie on 12/4/18.
//

#include <cstring>
#include <cuda_runtime.h>

#include "caffeine/common.hpp"
#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/blob.hpp"

#include "gtest/gtest.h"

namespace caffeine{

template <typename Dtype>
class NeuronLayerTest : public ::testing::Test{
protected:
    NeuronLayerTest()
    :blob_bottom_data_(new Blob<Dtype>(2, 3, 4, 5)),
    blob_top_data_(new Blob<Dtype>(2, 3, 4, 5)){
        FillerParameter filler_param_;
        GaussianFiller<Dtype> filler(filler_param_);
        filler.Fill(blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        blob_top_vec_.push_back(blob_top_data_);
    };
    virtual ~NeuronLayerTest(){
        delete blob_bottom_data_;
        delete blob_top_data_;
    };

    Blob<Dtype>* blob_bottom_data_;
    Blob<Dtype>* blob_top_data_;
    vector<Blob<Dtype>* > blob_bottom_vec_;
    vector<Blob<Dtype>* > blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NeuronLayerTest, Dtypes);

TYPED_TEST(NeuronLayerTest, TestReLU){
    LayerParameter layer_param;
    ReLULayer<TypeParam> layer(layer_param);
    layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_)); //TODO const

    const TypeParam* bottom_data = this->blob_bottom_data_->cpu_data();
    const TypeParam* top_data = this->blob_top_data_->cpu_data();
    const int count = this->blob_bottom_data_->count();
    for(int i=0; i<count; ++i){
        EXPECT_GE(top_data[i], 0.);
        EXPECT_TRUE(top_data[i] == 0 || bottom_data[i] == top_data[i]);
    }

}

} // namespace caffeine

