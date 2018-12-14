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
       GradientChecker(const Dtype stepsize, const Dtype threshold)
       :stepsize_(stepsize), threshold_(threshold){};

       void CheckGradient(Layer<Dtype>& layer, vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >& top, int check_bottom = -1);
   protected:
    Dtype GetObjAndGradient(vector<Blob<Dtype>* >& top);
    Dtype stepsize_;
    Dtype threshold_;
   };

}

#endif //BLACK_CAFFE_TEST_GRADIENT_CHECK_UTIL_HPP
