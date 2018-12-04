#ifndef CAFFEINE_COMMON_HPP_
#define CAFFEINE_COMMON_HPP_

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <cublas_v2.h>
#include <mkl_vsl.h>
#include "driver_types.h"

namespace caffeine {
    using boost::shared_ptr;

#define CUDA_CHECK(condition)       CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition)     CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define VSL_CHECK(condition)        CHECK_EQ((condition), VSL_STATUS_OK)


class Caffeine{
public:
    ~Caffeine();
    static Caffeine& Get();
    enum Brew {CPU, GPU};
    static cublasHandle_t cublas_handle();
    static VSLStreamStatePtr vsl_stream();
    static Brew mode();

    static void set_mode(Brew mode);

private:
    Caffeine();
    static shared_ptr<Caffeine> singleton_;
    cublasHandle_t cublas_handle_;
    VSLStreamStatePtr vsl_stream_;
    Brew mode_;

};

} // namespace caffeine


#endif  // CAFFEINE_COMMON_HPP_
