#include "dynet/nodes-const.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* Constant *************
string Constant::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constant(" << dim << ',' << value << ')';
  return s.str();
}

Dim Constant::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

template<class MyDevice>
void Constant::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 0, "Failed dimension check in Constant::forward");
  if (value == 0.f)
    TensorTools::zero(fx);
  else
    TensorTools::constant(fx, value);
}

template<class MyDevice>
void Constant::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("Called backward() on an arity 0 node");
}
DYNET_NODE_INST_DEV_IMPL(Constant)

}
