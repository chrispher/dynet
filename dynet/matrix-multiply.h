#ifndef DYNET_CUDA_MATRIX_MULTIPLY_H__
#define DYNET_CUDA_MATRIX_MULTIPLY_H__

#include "dynet/tensor.h"
#include "dynet/devices.h"
#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"

namespace dynet {

inline void MatrixMultiply(const Device_CPU & dev, const Tensor& l, const Tensor& r, Tensor& y, const float* acc_scalar) {

  y.tbvec().device(*dev.edevice) = *acc_scalar * y.tbvec();

  if(l.d.bd == 1 && r.d.bd == y.d.bd) {

      // If the left side has one batch, multiply by columns
      // [x, z, b] = [x, y] * [y, z, b]
      // -> [x, z*b] = [x, y], [y, z*b]
      y.colbatch_matrix().noalias() += *l * r.colbatch_matrix();

  } else {
    // Otherwise, loop over the batches
    DYNET_ARG_CHECK(r.d.bd != 1 || r.d.bd != l.d.bd,
                 "Number of batch elements in matrix multiply must match, but got: " << r.d.bd << ", " << l.d.bd);

    for(unsigned b = 0; b < y.d.bd; ++b)
      y.batch_matrix(b).noalias() += l.batch_matrix(b) * r.batch_matrix(b);

  }
}

inline void MatrixTranspMultiplyAcc(const dynet::Device_CPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  // computes l^T * r
  int max_b = std::max(l.d.bd, r.d.bd);
  if(l.d.bd == 1 && y.d.bd == r.d.bd) {
    y.colbatch_matrix().noalias() += (*l).transpose() * r.colbatch_matrix();
  } else {
    for(int b = 0; b < max_b; ++b)
      y.batch_matrix(b).noalias() += l.batch_matrix(b).transpose() * r.batch_matrix(b);
  }
}

inline void MatrixMultiplyTranspAcc(const dynet::Device_CPU & dev, const dynet::Tensor& l, const dynet::Tensor& r, dynet::Tensor& y) {
  int max_b = std::max(l.d.bd, r.d.bd);
  if(y.d.bd == 1 && (l.d.bd == r.d.bd)) {
    (*y).noalias() += l.colbatch_matrix() * r.colbatch_matrix().transpose();
  } else {
    for(int b = 0; b < max_b; ++b)
      y.batch_matrix(b).noalias() += l.batch_matrix(b) * r.batch_matrix(b).transpose();
  }
}


#endif
