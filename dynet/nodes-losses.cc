#include "dynet/nodes-losses.h"

#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* PairwiseRankLoss *************
string PairwiseRankLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "max(0, " << margin << " - " << arg_names[0] << " + " << arg_names[1] << ')';
  return os.str();
}

Dim PairwiseRankLoss::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2 &&
                          xs[0] == xs[1] &&
                          xs[0].rows() == 1 &&
                          (xs[0].ndims() == 1 || xs[0].ndims() == 2),
                          "Bad input dimensions in PairwiseRankLoss: " << xs);
  return xs[0].bd >= xs[1].bd ? xs[0] : xs[1];
}

template<class MyDevice>
void PairwiseRankLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().binaryExpr(xs[1]->tvec(), FPairwiseRankLoss(margin));
}

template<class MyDevice>
void PairwiseRankLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  if (i == 0) {
    dEdxi.tvec().device(*dev.edevice) -= fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
  } else {
    dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FRectifyBackward());
  }
}
DYNET_NODE_INST_DEV_IMPL(PairwiseRankLoss)

// ************* BinaryLogLoss *************
string BinaryLogLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "binary_log_loss(" << arg_names[0] << ", " << arg_names[1] << ')';
  return os.str();
}

Dim BinaryLogLoss::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in BinaryLogLoss")
  DYNET_ARG_CHECK(xs[0].single_batch() == xs[1].single_batch(), "Bad input dimensions in BinaryLogLoss: " << xs);
  DYNET_ARG_CHECK(xs[0].bd == xs[1].bd, "BinaryLogLoss with unmatched batches is not implemented yet (pull requests welcome): " << xs);
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

template<class MyDevice>
void BinaryLogLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.t<0>().device(*dev.edevice) = xs[0]->tvec().binaryExpr(xs[1]->tvec(), FBinaryLogLoss()).sum();
}

template<class MyDevice>
void BinaryLogLoss::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[i]->tvec().binaryExpr(xs[1-i]->tvec(), FBinaryLogLossBackward(as_scalar(dEdf)));
}
DYNET_NODE_INST_DEV_IMPL(BinaryLogLoss)

// ************* PoissonRegressionLoss *************
string PoissonRegressionLoss::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "-log Poisson(" << pty << "; lambda=\\exp" << arg_names[0] << ')';
  return s.str();
}

Dim PoissonRegressionLoss::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1 && xs[0].size() == 1, "Bad input dimensions in PoissonRegressionLoss: " << xs);
  return xs[0];
}

template<class MyDevice>
void PoissonRegressionLoss::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  const real y = *pty;
  const auto z = std::lgamma(y + 1);
  // const auto x = as_scalar(*xs[0]);
  fx.t<0>().device(*dev.edevice) = xs[0]->t<0>().exp() + z - xs[0]->t<0>() * y;
}

template<class MyDevice>
void PoissonRegressionLoss::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  const real y = *pty;
  dEdxi.t<0>().device(*dev.edevice) += xs[0]->t<0>().exp() - y;
}
DYNET_NODE_INST_DEV_IMPL(PoissonRegressionLoss)

}
