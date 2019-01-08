//
// Created by chujie on 12/21/18.
//

#ifndef BLACK_CAFFE_GEMM_HPP
#define BLACK_CAFFE_GEMM_HPP
#include <mkl.h>
#include <cublas_v2.h>

namespace caffeine {

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
    template <typename Dtype>
    inline void decaf_cpu_gemm(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                               const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                               Dtype* C);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
    template <typename Dtype>
    void decaf_gpu_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                        const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                        Dtype* C);

    template <typename Dtype>
    void decaf_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                        Dtype* y);

    template <typename Dtype>
    void decaf_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                        Dtype* y);

}  // namespace caffeine

#endif //BLACK_CAFFE_GEMM_HPP
