#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include <algorithm>

using std::max;

namespace caffeine{

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, vector<Blob<Dtype>* >* top){
    const Dtype* data = bottom[0]->cpu_data(); // TODO bottom[0].cpudata()
    const int count = bottom[0]->count();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    for(int i=0; i<count; ++i){
        top_data[i] = max(data[i], Dtype(0));
    }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top, const bool propagate_down, vector<Blob<Dtype>* >* bottom){
    if (propagate_down){
        const Dtype* bottom_data = (*bottom)[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
        const int count =top[0]->count();
        for (int i = 0; i < count; ++i) {
            bottom_diff[i] = top_diff[i] * (bottom_data[i] >= 0);
        }
    }
    return Dtype(0);
}

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        out[index] = max(in[index], Dtype(0.));
    }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = (*top)[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
//    const int blocks = (count + CAFFEINE_CUDA_NUM_THREADS - 1) /
//                       CAFFEINE_CUDA_NUM_THREADS;
    ReLUForward<Dtype><<<CAFFEINE_GET_BLOCKS(count), CAFFEINE_CUDA_NUM_THREADS>>>(count, bottom_data,
            top_data);
}

    template <typename Dtype>
    __global__ void ReLUBackward(const int n, const Dtype* in_diff,
                                 const Dtype* in_data, Dtype* out_diff) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < n) {
            out_diff[index] = in_diff[index] * (in_data[index] >= 0);
        }
    }

    template <typename Dtype>
    Dtype ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const bool propagate_down,
                                         vector<Blob<Dtype>*>* bottom) {
        if (propagate_down) {
            const Dtype* bottom_data = (*bottom)[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
            const int count = (*bottom)[0]->count();
//            const int blocks = (count + CAFFEINE_CUDA_NUM_THREADS - 1) /
//                               CAFFEINE_CUDA_NUM_THREADS;
            ReLUBackward<Dtype><<<CAFFEINE_GET_BLOCKS(count), CAFFEINE_CUDA_NUM_THREADS>>>(count, top_diff,
                    bottom_data, bottom_diff);
        }
        return Dtype(0);
    }

    template class ReLULayer<float>;
    template class ReLULayer<double>;

}// namespace caffeine