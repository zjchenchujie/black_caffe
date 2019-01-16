//
// Created by chujie on 12/21/18.
//

#ifndef BLACK_CAFFE_GEMM_HPP
#define BLACK_CAFFE_GEMM_HPP
#include <mkl.h>
#include <cublas_v2.h>

namespace caffe {

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
    template <typename Dtype>
    inline void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                               const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                               Dtype* C);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
    template <typename Dtype>
    void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                        const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                        Dtype* C);

    template <typename Dtype>
    void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                        Dtype* y);

    template <typename Dtype>
    void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                        Dtype* y);

    template <typename Dtype>
    void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
                       Dtype* Y);

    template <typename Dtype>
    void caffe_copy(const int N, const Dtype *X, Dtype *Y);

    template <typename Dtype>
    void caffe_sqr(const int N, const Dtype* a, Dtype* y);

    template <typename Dtype>
    void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

    template <typename Dtype>
    void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

    template <typename Dtype>
    void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);


}  // namespace caffe

#endif //BLACK_CAFFE_GEMM_HPP
