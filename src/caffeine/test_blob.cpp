#include "caffeine/blob.hpp"

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
//    EXPECT_TRUE(this->blob_->cpu_data());
//    EXPECT_TRUE(this->blob_->gpu_data() == NULL);
//    EXPECT_TRUE(this->blob_->mutable_cpu_data() == NULL);
//    EXPECT_TRUE(this->blob_->mutable_gpu_data() == NULL);
    EXPECT_TRUE(this->blob_preshaped_->cpu_data());
    EXPECT_TRUE(this->blob_preshaped_->gpu_data());
    EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
    EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());

}
TYPED_TEST(BlobSimpleTest, TestInitialization){
    EXPECT_TRUE(this->blob_);
    EXPECT_TRUE(this->blob_preshaped_);
    EXPECT_EQ(0, this->blob_->num());
    EXPECT_EQ(0, this->blob_->channels());
    EXPECT_EQ(0, this->blob_->height());
    EXPECT_EQ(0, this->blob_->width());
    EXPECT_EQ(0, this->blob_->count());
    EXPECT_EQ(2, this->blob_preshaped_->num());
    EXPECT_EQ(3, this->blob_preshaped_->channels());
    EXPECT_EQ(4, this->blob_preshaped_->height());
    EXPECT_EQ(5, this->blob_preshaped_->width());
    EXPECT_EQ(120, this->blob_preshaped_->count());
}

TYPED_TEST(BlobSimpleTest, TestReshape){
    this->blob_->Reshape(2,3,4,5);
    EXPECT_EQ(2, this->blob_->num());
    EXPECT_EQ(3, this->blob_->channels());
    EXPECT_EQ(4, this->blob_->height());
    EXPECT_EQ(5, this->blob_->width());
    EXPECT_EQ(120, this->blob_->count());

}

}