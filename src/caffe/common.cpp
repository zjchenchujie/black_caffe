//
// Created by chujie on 12/2/18.
//
#include "caffe/common.hpp"

//#include <boost/shared_ptr.hpp>

namespace caffe{

    shared_ptr<caffe> caffe::singleton_;
    caffe::caffe()
    :mode_(caffe::CPU), phase_(caffe::TRAIN){
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CURAND_CHECK(curandCreateGenerator(&curand_generator_,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_,
                                                        1701ULL));
        VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
    }

    caffe::~caffe() {
        if(!cublas_handle_){
           CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        }
        if(!vsl_stream_){
            VSL_CHECK(vslDeleteStream(&vsl_stream_));
        }

        if(!curand_generator_){
            CURAND_CHECK(curandDestroyGenerator(curand_generator_));
        }
    }

    caffe& caffe::Get(){
        if(!singleton_){
            singleton_.reset(new caffe());
        }
        return *singleton_;
    }

    caffe::Brew caffe::mode() {
        return Get().mode_;
    }

    caffe::Phase caffe::phase() {
        return Get().phase_;
    }

    cublasHandle_t caffe::cublas_handle() {
        return Get().cublas_handle_;
    }

    curandGenerator_t caffe::curand_generator() {
        return Get().curand_generator_;
    }

    VSLStreamStatePtr caffe::vsl_stream() {
        return Get().vsl_stream_;
    }

    void caffe::set_mode(caffe::Brew mode) {
        Get().mode_ = mode;
    }

    void caffe::set_phase(caffe::Phase phase) {
        Get().phase_ = phase;
    }

    void caffe::set_random_seed(unsigned int seed) {
        CURAND_CHECK(curandDestroyGenerator(curand_generator()));
        CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
                                                        (unsigned long long)seed));
        VSL_CHECK(vslDeleteStream(&(Get().vsl_stream_)));
        VSL_CHECK(vslNewStream(&(Get().vsl_stream_), VSL_BRNG_MT19937, seed));
    }


} // namespace caffe

