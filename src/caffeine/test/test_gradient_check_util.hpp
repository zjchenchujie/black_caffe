//
// Created by chujie on 12/14/18.
//

#ifndef BLACK_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP
#define BLACK_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP

#include "caffeine/layer.hpp"

namespace caffeine {

   template <typename Dtype>
   class GradientChecker{
   public:
       GradientChecker(const Dtype stepsize, const Dtype threshold, const unsigned int seed = 1701, const Dtype kink = 0., const Dtype kink_range = -1)
       :stepsize_(stepsize), threshold_(threshold), seed_(seed), kink_(kink), kink_range_(kink_range){};

       void CheckGradient(Layer<Dtype>& layer, vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >& top,
               int check_bottom = -1){
          layer.SetUp(bottom, &top);
          CheckGradientSingle(layer, bottom, top, check_bottom, -1,-1);
       }

       void CheckGradientExhaustive(Layer<Dtype>& layer,
                                    vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top,
                                    int check_bottom = -1);

       void CheckGradientSingle(Layer<Dtype>& layer, vector<Blob<Dtype>*>& bottom,
                                vector<Blob<Dtype>*>& top, int check_bottom, int top_id,
                                int top_data_id);
   protected:
    Dtype GetObjAndGradient(vector<Blob<Dtype>* >& top, int top_id = -1, int top_data_id = -1);
    Dtype stepsize_;
    Dtype threshold_;
    unsigned int seed_;
    Dtype kink_;
    Dtype kink_range_;
   };

}

#endif //BLACK_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP
