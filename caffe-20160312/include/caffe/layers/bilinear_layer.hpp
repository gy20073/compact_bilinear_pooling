#ifndef CAFFE_BILINEAR_LAYER_HPP_
#define CAFFE_BILINEAR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes @f$ y = \sum_p flatten(x_{1p} x_{2p}^T) @f$,
 * where p is the spatial index. The "flatten" operation
 * convert a matrix to a vector.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C_1 \times H \times W) @f$
 *      the inputs @f$ x_1 @f$
 *   -# @f$ (N \times C_2 \times H \times W) @f$
 *      the inputs @f$ x_2 @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C_1C_2 \times 1 \times 1) @f$
 *      the bilinear pooled vector @f$ y @f$
 */
template<typename Dtype>
class BilinearLayer: public Layer<Dtype> {
 public:
    explicit BilinearLayer(const LayerParameter& param) :
            Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "Bilinear";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }

 protected:
    /// @copydoc BilinearLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    /**
     * @brief Computes the error gradient w.r.t. the bilinear inputs.
     *
     * @param top output Blob vector (length 1), providing the error gradient with
     *      respect to the outputs
     *   -# @f$ (N \times C \times 1 \times 1) @f$
     *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
     *      with respect to computed outputs @f$ y @f$
     * @param propagate_down see Layer::Backward.
     * @param bottom input Blob vector (length 2)
     *   -# @f$ (N \times C_1/C_2 \times H \times W) @f$
     *      the inputs @f$ x_1/x_2 @f$; Backward fills their diff with
     *      gradients if propagate_down[0]
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_BILINEAR_LAYER_HPP_
