//
// Created by chujie on 12/8/18.
//

#include "caffeine/vision_layers.hpp"
#include "caffeine/blob.hpp"
#include "caffeine/filler.hpp"
#include "gtest/gtest.h"
namespace caffeine{


template <typename Dtype>
class InnerProductLayerTest : public ::testing::Test{
protected:
    InnerProductLayerTest()
    :bottom_data_(new Blob<Dtype>(2, 3, 4, 5)),
    top_data_(new Blob<Dtype>()) {
        FillerParameter filler_param;
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->bottom_data_);
        bottom_data_vec_.push_back(bottom_data_);
        top_data_vec_.push_back(top_data_);

    };
    virtual ~InnerProductLayerTest(){
        delete bottom_data_;
        delete top_data_;
    }

    Blob<Dtype>* const bottom_data_;
    Blob<Dtype>* const top_data_;
    vector<Blob<Dtype>* > bottom_data_vec_;
    vector<Blob<Dtype>* > top_data_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(InnerProductLayerTest, Dtypes);

TYPED_TEST(InnerProductLayerTest, TestSetup){
    LayerParameter layer_param;
    layer_param.set_num_output(10);
    shared_ptr<InnerProductLayer<TypeParam> > layer(
            new InnerProductLayer<TypeParam>(layer_param));
    layer->SetUp(this->bottom_data_vec_, &(this->top_data_vec_));
    EXPECT_EQ((this->top_data_vec_[0])->num(), 2);
    EXPECT_EQ((this->top_data_vec_[0])->height(), 1);
    EXPECT_EQ((this->top_data_vec_[0])->width(), 1);
    EXPECT_EQ((this->top_data_vec_[0])->channels(), 10);
    layer_param.set_gemm_last_dim(true);
    layer.reset(new InnerProductLayer<TypeParam >(layer_param));
    layer->SetUp(this->bottom_data_vec_, &(this->top_data_vec_));
    EXPECT_EQ((this->top_data_vec_[0])->num(), 2);
    EXPECT_EQ((this->top_data_vec_[0])->height(), 3);
    EXPECT_EQ((this->top_data_vec_[0])->width(), 4);
    EXPECT_EQ((this->top_data_vec_[0])->channels(), 10);
}

TYPED_TEST(InnerProductLayerTest, TestForwardCPU){
    LayerParameter layer_param;
    Caffeine::set_mode(Caffeine::CPU);
    layer_param.set_num_output(10);
    layer_param.mutable_weight_filler()->set_type("uniform");
    layer_param.mutable_bias_filler()->set_type("uniform");
    layer_param.mutable_bias_filler()->set_min(1);
    layer_param.mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<TypeParam> > layer(
            new InnerProductLayer<TypeParam>(layer_param));
    layer->SetUp(this->bottom_data_vec_, &(this->top_data_vec_));
    layer->Forward(this->bottom_data_vec_, &(this->top_data_vec_));
    const TypeParam *data = this->top_data_vec_[0]->cpu_data();
    const int count = this->top_data_vec_[0]->count();
    for (int i=0; i<count;++i){
        EXPECT_GE(data[i], 1.);
    }
}

//    TYPED_TEST(InnerProductLayerTest, TestGPU) {
//        if (sizeof(TypeParam) == 4 || CAFFEINE_TEST_CUDA_PROP.major >= 2) {
//            LayerParameter layer_param;
//            Caffeine::set_mode(Caffeine::GPU);
//            layer_param.set_num_output(10);
//            layer_param.mutable_weight_filler()->set_type("uniform");
//            layer_param.mutable_bias_filler()->set_type("uniform");
//            layer_param.mutable_bias_filler()->set_min(1);
//            layer_param.mutable_bias_filler()->set_max(2);
//            shared_ptr<InnerProductLayer<TypeParam> > layer(
//                    new InnerProductLayer<TypeParam>(layer_param));
//            layer->SetUp(this->bottom_data_vec_, &(this->top_data_vec_));
//            layer->Forward(this->bottom_data_vec_, &(this->top_data_vec_));
//            const TypeParam* data = this->top_data_->cpu_data();
//            const int count = this->top_data_->count();
//            for (int i = 0; i < count; ++i) {
//                EXPECT_GE(data[i], 1.);
//            }
//        } else {
//            LOG(ERROR) << "Skipping test due to old architecture.";
//        }
//    }

} // namespace caffeine

