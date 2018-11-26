//
// Created by chujie on 18-11-22.
//

#ifndef BLACK_CAFFE_COMMON_H
#define BLACK_CAFFE_COMMON_H

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include "driver_types.h"

namespace caffeine {
    using boost::shared_ptr;
}

static std::ostream nullout(0);

//#define CUDA_LOG_IF(condition) \
//    ((condition) == cudaSuccess) ? nullout : std::cout
#define CUDA_CHECK(condition) \
    CHECK((condition)==cudaSuccess)

#endif //BLACK_CAFFE_COMMON_H
