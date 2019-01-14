//
// Created by chujie on 12/4/18.
//

#include <cstring>
#include <cuda_runtime.h>

#include "caffeine/filler.hpp"
#include "gtest/gtest.h"

#include "caffeine/test/test_caffeine_main.hpp"

namespace caffeine {

typedef ::testing::Types<float, double> Dtype;

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test{
protected:
    ConstantFillerTest()
    :blob_(new Blob<Dtype>(2, 3, 4, 5)),
    filler_parameter_(){
        filler_parameter_.set_value(10.);
        filler_.reset(new ConstantFiller<Dtype>(filler_parameter_));
        filler_->Fill(blob_);
    };
    virtual ~ConstantFillerTest(){ delete blob_; };
    Blob<Dtype>* const blob_;
    FillerParameter filler_parameter_;
    shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, Dtype);
TYPED_TEST(ConstantFillerTest, TestFill){
    EXPECT_TRUE(this->blob_);
    const TypeParam *data = this->blob_->cpu_data();
    const int count = this->blob_->count();
    for(int i=0; i<count; i++){
        EXPECT_GE(data[i], this->filler_parameter_.value());
    }
}

template <typename Dtype>
class UniformFillerTest : public ::testing::Test{
protected:
    UniformFillerTest()
    :blob_(new Blob<Dtype>(2, 3, 4, 5)),
     filler_parameter_(){
        filler_parameter_.set_min(1.);
        filler_parameter_.set_max(2.);
        filler_.reset(new UniformFiller<Dtype>(filler_parameter_));
        filler_->Fill(blob_);
    };
    virtual ~UniformFillerTest(){ delete blob_; };

    Blob<Dtype>* const blob_;
    FillerParameter filler_parameter_;
    shared_ptr<UniformFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, Dtype);
TYPED_TEST(UniformFillerTest, TestUniformFiller){
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const TypeParam *data = this->blob_->cpu_data();
    for (int i=0; i<count; ++i){
        EXPECT_GE(data[i], this->filler_parameter_.min());
        EXPECT_LE(data[i], this->filler_parameter_.max());
    }
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test{
protected:
    GaussianFillerTest()
    :blob_(new Blob<Dtype>(2, 3, 4, 5)),
    filler_parameter_(){
        filler_parameter_.set_mean(10.);
        filler_parameter_.set_std(0.1);
        filler_.reset(new GaussianFiller<Dtype>(filler_parameter_));
        filler_->Fill(blob_);
    };
    virtual ~GaussianFillerTest(){ delete blob_; };

    Blob<Dtype>* const blob_;
    FillerParameter filler_parameter_;
    shared_ptr<GaussianFiller<Dtype> > filler_;

};

TYPED_TEST_CASE(GaussianFillerTest, Dtype);
    TYPED_TEST(GaussianFillerTest, TestFill) {
        EXPECT_TRUE(this->blob_);
        const int count = this->blob_->count();
        const TypeParam* data = this->blob_->cpu_data();
        TypeParam mean = 0.;
        TypeParam var = 0.;
        for (int i = 0; i < count; ++i) {
            mean += data[i];
            var += (data[i] - this->filler_parameter_.mean()) *
                   (data[i] - this->filler_parameter_.mean());
        }
        mean /= count;
        var /= count;
        // Very loose test.
        EXPECT_GE(mean, this->filler_parameter_.mean() - this->filler_parameter_.std() * 5);
        EXPECT_LE(mean, this->filler_parameter_.mean() + this->filler_parameter_.std() * 5);
        TypeParam target_var = this->filler_parameter_.std() * this->filler_parameter_.std();
        EXPECT_GE(var, target_var / 5.);
        EXPECT_LE(var, target_var * 5.);
    }


}// namespace caffeine

