#include "caffeine/test/test_gradient_check_util.hpp"
#include <cmath>
#include "gtest/gtest.h"
#include <glog/logging.h>

using std::max;

namespace caffeine {

    template <typename Dtype>
    void GradientChecker<Dtype>::CheckGradientSingle(caffeine::Layer<Dtype> &layer,
                                                     vector<caffeine::Blob<Dtype> *> &bottom,
                                                     vector<caffeine::Blob<Dtype> *> &top, int check_bottom, int top_id,
                                                     int top_data_id) {
        layer.SetUp(bottom, &top);
        // First, figure out what blobs we need to check against.
        vector<Blob<Dtype>*> blobs_to_check;
        for (int i = 0; i < layer.params().size(); ++i) {
            blobs_to_check.push_back(&layer.params()[i]);
        }
        if (check_bottom < 0) {
            for (int i = 0; i < bottom.size(); ++i) {
                blobs_to_check.push_back(bottom[i]);
            }
        } else {
            CHECK(check_bottom < bottom.size());
            blobs_to_check.push_back(bottom[check_bottom]);
        }
        // go through the bottom and parameter blobs
//        LOG(ERROR) << "Checking " << blobs_to_check.size() << " blobs.";
        for (int blobid = 0; blobid < blobs_to_check.size(); ++blobid) {
            Blob<Dtype>* current_blob = blobs_to_check[blobid];
//            LOG(ERROR) << "Blob " << blobid << ": checking " << current_blob->count() << " parameters.";
            // go through the values
            for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
                // First, obtain the original data
                Caffeine::set_random_seed(seed_);
                layer.Forward(bottom, &top);
                Dtype computed_objective = GetObjAndGradient(top, top_id, top_data_id);
                // Get any additional loss from the layer
                computed_objective += layer.Backward(top, true, &bottom);
                Dtype computed_gradient = current_blob->cpu_diff()[feat_id];
                // compute score by adding stepsize
                current_blob->mutable_cpu_data()[feat_id] += stepsize_;
                Caffeine::set_random_seed(seed_);
                layer.Forward(bottom, &top);
                Dtype positive_objective = GetObjAndGradient(top, top_id, top_data_id);
                positive_objective += layer.Backward(top, true, &bottom);
                // compute score by subtracting stepsize
                current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
                Caffeine::set_random_seed(seed_);
                layer.Forward(bottom, &top);
                Dtype negative_objective = GetObjAndGradient(top, top_id, top_data_id);
                negative_objective += layer.Backward(top, true, &bottom);
                // Recover stepsize
                current_blob->mutable_cpu_data()[feat_id] += stepsize_;
                Dtype estimated_gradient = (positive_objective - negative_objective) /
                                           stepsize_ / 2.;
                Dtype feature = current_blob->cpu_data()[feat_id];
                //LOG(ERROR) << "debug: " << current_blob->cpu_data()[feat_id] << " "
                //    << current_blob->cpu_diff()[feat_id];
                if (kink_ - kink_range_ > feature || feature > kink_ + kink_range_) {
                    // We check relative accuracy, but for too small values, we threshold
                    // the scale factor by 1.
                    Dtype scale = max(max(fabs(computed_gradient), fabs(estimated_gradient)),
                                      1.);
                    EXPECT_GT(computed_gradient, estimated_gradient - threshold_ * scale);
                    EXPECT_LT(computed_gradient, estimated_gradient + threshold_ * scale);
                }
                //LOG(ERROR) << "Feature: " << current_blob->cpu_data()[feat_id];
                //LOG(ERROR) << "computed gradient: " << computed_gradient
                //    << " estimated_gradient: " << estimated_gradient;
            }
        }
    }

    template <typename Dtype>
    void GradientChecker<Dtype>::CheckGradientExhaustive(caffeine::Layer<Dtype> &layer,
                                                         vector<caffeine::Blob<Dtype> *> &bottom,
                                                         vector<caffeine::Blob<Dtype> *> &top, int check_bottom) {
        layer.SetUp(bottom, &top);
        for(int i = 0 ; i < top.size(); i++){
            for(int j = 0; j < top[i]->count(); j++){
                CheckGradientSingle(layer, bottom, top, check_bottom, i, j);
            }
        }
    }

    template <typename Dtype>
    Dtype GradientChecker<Dtype>::GetObjAndGradient(vector<Blob<Dtype>*>& top, int top_id, int top_data_id) {
        Dtype loss = 0;
        if(top_id < 0){
            // the loss will be half of the sum of squares of all outputs
            for (int i = 0; i < top.size(); ++i) {
                Blob<Dtype>* top_blob = top[i];
                const Dtype* top_blob_data = top_blob->cpu_data();
                Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
                int count = top_blob->count();
                for (int j = 0; j < count; ++j) {
                    loss += top_blob_data[j] * top_blob_data[j];
                }
                // set the diff: simply the data.
                memcpy(top_blob_diff, top_blob_data, sizeof(Dtype) * count);
            }
            loss /= 2.;
        } else{
            // the loss will be the top_data_id-th element in the top_id-th blob.
            for (int i = 0; i < top.size(); ++i) {
                Blob<Dtype>* top_blob = top[i];
                Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
                memset(top_blob_diff, 0, sizeof(Dtype) * top_blob->count());
            }
            loss = top[top_id]->cpu_data()[top_data_id];
            top[top_id]->mutable_cpu_diff()[top_data_id] = 1.;
        }

        return loss;
    }

    INSTANTIATE_CLASS(GradientChecker);

}


