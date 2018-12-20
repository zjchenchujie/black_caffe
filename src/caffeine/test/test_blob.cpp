#include "caffeine/blob.hpp"
#include "caffeine/filler.hpp"

#include <gtest/gtest.h>
namespace caffeine{

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test{
protected:
    BlobSimpleTest()
        :blob_(new Blob<Dtype>()),
         blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {};
    virtual ~BlobSimpleTest(){delete blob_; delete blob_preshaped_;}
    Blob<Dtype>* const blob_;
    Blob<Dtype>* const blob_preshaped_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(BlobSimpleTest, Dtypes);
TYPED_TEST(BlobSimpleTest, TestPointer){

    EXPECT_TRUE(this->blob_preshaped_->cpu_data());
    EXPECT_TRUE(this->blob_preshaped_->gpu_data());
    EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
    EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());

}
TYPED_TEST(BlobSimpleTest, TestInitialization){
    EXPECT_TRUE(this->blob_);
    EXPECT_TRUE(this->blob_preshaped_);
    EXPECT_EQ(0, this->blob_->num());
    EXPECT_EQ(0, this->blob_->height());
    EXPECT_EQ(0, this->blob_->width());
    EXPECT_EQ(0, this->blob_->channels());
    EXPECT_EQ(0, this->blob_->count());
    EXPECT_EQ(2, this->blob_preshaped_->num());
    EXPECT_EQ(3, this->blob_preshaped_->channels());
    EXPECT_EQ(4, this->blob_preshaped_->height());
    EXPECT_EQ(5, this->blob_preshaped_->width());
    EXPECT_EQ(120, this->blob_preshaped_->count());
}

TYPED_TEST(BlobSimpleTest, TestReshape){
    this->blob_->Reshape(2,3,4,5);
    EXPECT_EQ(2, this->blob_preshaped_->num());
    EXPECT_EQ(3, this->blob_preshaped_->channels());
    EXPECT_EQ(4, this->blob_preshaped_->height());
    EXPECT_EQ(5, this->blob_preshaped_->width());
    EXPECT_EQ(120, this->blob_->count());

}

TYPED_TEST(BlobSimpleTest, TestSourceConstructor){
    Blob<TypeParam> source(2, 3, 4, 5);
    FillerParameter filler_parameter;
    UniformFiller<TypeParam > filler(filler_parameter);
    filler.Fill(&source);
    Blob<TypeParam> target(source);
    const int count = source.count();
    const TypeParam *source_data = source.cpu_data();
    const TypeParam *target_data = target.cpu_data();
    EXPECT_EQ(target.num(), source.num());
    EXPECT_EQ(target.channels(), source.channels());
    EXPECT_EQ(target.height(),source.height());
    EXPECT_EQ(target.width(), source.width());
    for(int i=0 ; i<count; ++i){
        EXPECT_EQ(source_data[i], target_data[i]);
    }
}

}