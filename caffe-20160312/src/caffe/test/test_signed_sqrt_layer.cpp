#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/signed_sqrt_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class SignedSqrtLayerTest: public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

 protected:
    SignedSqrtLayerTest() :
            blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
            blob_top_(new Blob<Dtype>()) {}
    virtual void SetUp() {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_0_);
        // try to avoid testing in the unstable region
        caffe_add_scalar(blob_bottom_0_->count(), Dtype(3.0),
                         blob_bottom_0_->mutable_cpu_data());

        blob_bottom_vec_.push_back(blob_bottom_0_);
        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~SignedSqrtLayerTest() {
        delete blob_bottom_0_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_0_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SignedSqrtLayerTest, TestDtypesAndDevices);

TYPED_TEST(SignedSqrtLayerTest, TestGradientOutplace) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SignedSqrtLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-1);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

}  // namespace caffe
