//
// Created by chujie on 12/2/18.
//
#include "caffe/common.hpp"

//#include <boost/shared_ptr.hpp>

namespace caffe{

    shared_ptr<Caffe> Caffe::singleton_;
    Caffe::Caffe()
    :mode_(Caffe::CPU), phase_(Caffe::TRAIN){
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CURAND_CHECK(curandCreateGenerator(&curand_generator_,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_,
                                                        1701ULL));
        VSL_CHECK(vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701));
    }

    Caffe::~Caffe() {
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

    Caffe& Caffe::Get(){
        if(!singleton_){
            singleton_.reset(new Caffe());
        }
        return *singleton_;
    }

    Caffe::Brew Caffe::mode() {
        return Get().mode_;
    }

    Caffe::Phase Caffe::phase() {
        return Get().phase_;
    }

    cublasHandle_t Caffe::cublas_handle() {
        return Get().cublas_handle_;
    }

    curandGenerator_t Caffe::curand_generator() {
        return Get().curand_generator_;
    }

    VSLStreamStatePtr Caffe::vsl_stream() {
        return Get().vsl_stream_;
    }

    void Caffe::set_mode(Caffe::Brew mode) {
        Get().mode_ = mode;
    }

    void Caffe::set_phase(Caffe::Phase phase) {
        Get().phase_ = phase;
    }

    void Caffe::set_random_seed(unsigned int seed) {
        CURAND_CHECK(curandDestroyGenerator(curand_generator()));
        CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
                                                        (unsigned long long)seed));
        VSL_CHECK(vslDeleteStream(&(Get().vsl_stream_)));
        VSL_CHECK(vslNewStream(&(Get().vsl_stream_), VSL_BRNG_MT19937, seed));
    }


} // namespace caffe

