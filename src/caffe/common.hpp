#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <cublas_v2.h>
#include <mkl_vsl.h>
#include <cuda.h>
#include <curand.h>
#include "driver_types.h"

#define CUDA_CHECK(condition)       CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition)     CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define CURAND_CHECK(condition)     CHECK_EQ((condition), CURAND_STATUS_SUCCESS)
#define VSL_CHECK(condition)        CHECK_EQ((condition), VSL_STATUS_OK)

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

#define CUDA_POST_KERNEL_CHECK \
  if (cudaSuccess != cudaPeekAtLastError()) {\
    LOG(FATAL) << "Cuda kernel failed. Error: " << cudaGetLastError(); \
  }

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

namespace caffe {
using boost::shared_ptr;

const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N){
    return (N + CAFFE_CUDA_NUM_THREADS -1) / CAFFE_CUDA_NUM_THREADS;
}

class Caffe{
public:
    ~Caffe();
    static Caffe& Get();
    enum Brew {CPU, GPU};
    enum Phase {TRAIN, TEST};

    // The getters for the variables.
    // Returns the cublas handle.
    static cublasHandle_t cublas_handle();
    // Returns the curand generator
    static curandGenerator_t curand_generator();
    // Returns the MKL random stream
    static VSLStreamStatePtr vsl_stream();
    // Returns the working mode: working on CPU or GPU
    static Brew mode();
    // Returns the working phase: Training or TEST
    static Phase phase();

    // The setter for variables
    // Set the working mode: CPU or GPU
    static void set_mode(Brew mode);
    // set the working phase: TRAIN or TEST
    static void set_phase(Phase phase);
    // set the random seed for both MKL or curand
    static void set_random_seed(unsigned int seed);

private:
    Caffe();
    static shared_ptr<Caffe> singleton_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_generator_;
    VSLStreamStatePtr vsl_stream_;
    Brew mode_;
    Phase phase_;

};

} // namespace caffe


#endif  // caffe_COMMON_HPP_
