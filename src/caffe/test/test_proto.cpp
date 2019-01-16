//
// Created by chujie on 1/15/19.
//
#include <string>
#include <google/protobuf/text_format.h>
#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/proto/layer_param.pb.h"

namespace caffe{

class ProtoTest: public ::testing::Test{

};

TEST_F(ProtoTest, TestSerialization){
    LayerParameter param;
    param.set_name("test");
    param.set_type("dummy");
    std::cout << "Printing in binary format. "<< std::endl;
    std::cout << param.SerializeAsString() << std::endl;
    std::cout << "Printing in text format. " << std::endl;
    std::string str;
    google::protobuf::TextFormat::PrintToString(param, &str);
    std::cout << str << std::endl;
    EXPECT_TRUE(true);
}

}
