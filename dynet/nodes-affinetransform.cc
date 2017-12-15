#include "dynet/nodes-affinetransform.h"

#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {

// ************* AffineTransform *************
string AffineTransform::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Dim AffineTransform::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK((xs.size() - 1) % 2 == 0, "Bad number of inputs in AffineTransform: " << xs);
  if(xs.size() == 1) return xs[0];
  DYNET_ARG_CHECK(xs[0].rows() == xs[1].rows() && xs[1].cols() == xs[2].rows(),
                          "Bad dimensions for AffineTransform: " << xs);
  Dim d = (xs[2].cols() != 1 ?
           Dim({xs[0].rows(), xs[2].cols()}, max(max(xs[0].bd, xs[1].bd), xs[2].bd)) :
           Dim({xs[0].rows()}, max(max(xs[0].bd, xs[1].bd), xs[2].bd)));
  for (unsigned i = 3; i < xs.size(); i += 2) {
    DYNET_ARG_CHECK(xs[i].cols() == xs[i+1].rows() && d.rows() == xs[i].rows() && d.cols() == xs[i+1].cols(),
                            "Bad dimensions for AffineTransform: " << xs);
    d.bd = max(max(d.bd, xs[i].bd), xs[i+1].bd);
  }
  return d;
}

int AffineTransform::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::affine);
  // This is a heuristic: we assume that we often have "b + W * x" shaped affine transforms
  // so when everything is batch size one, optimize for this case
  if(dim.bd == 1) {
    s.add_node(args[0]);
    for(size_t i = 1; i < args.size(); i += 2) {
      s.add_node(args[i]);
      s.add_dim(cg.nodes[args[i+1]]->dim); // TODO: this is not the exact same as dim->print_profile
    }
  } else {
    for(auto nid : args) {
      const Dim & d = cg.nodes[nid]->dim;
      if(d.bd == 1)
        s.add_node(nid);
      else
        s.add_dim(d); // TODO: this is not the exact same as dim->print_profile
    }
  }
  return sm.get_idx(s);
}

std::vector<int> AffineTransform::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if(dim.bd == 1) {
    for(size_t i = 2; i < ret.size(); i += 2)
      ret[i] = 1;
  } else {
    for(size_t i = 0; i < ret.size(); ++i)
      ret[i] = (cg.nodes[args[i]]->dim.bd > 1);
  }
  return ret;
}

// Affine transform uses different implementations for CPU and GPU because this is 
// much faster than using Eigen's tensor contractions (as of the writing)
template<class MyDevice>
void AffineTransform::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() % 2 == 1, "Failed dimension check in AffineTransform::forward");
  if (xs.size() == 1) {
    fx.tvec().device(*dev.edevice) = xs[0]->tvec();
    return;
  } else {
    // Add the first matrix
    size_t b_size = xs[0]->d.size(), fx_size = fx.d.size();
    if(fx_size == b_size) {
      fx.tvec().device(*dev.edevice) = xs[0]->tvec();
    } else {
      DYNET_ARG_CHECK(xs[0]->d.bd == 1, "In AffineTransform, broadcasting over columns with mini-batched inputs is not implemented yet");
      float *curr_ptr = fx.v, *end_ptr = curr_ptr + fx.d.size(), *in_ptr = xs[0]->v;
      do {
        memcpy(curr_ptr, in_ptr, sizeof(float)*b_size);
        curr_ptr += b_size;
      } while(curr_ptr != end_ptr);
    }

    // Perform multiplication
    for (unsigned i = 1; i < xs.size(); i += 2) {
      if(xs[i]->d.bd == 1 && xs[i+1]->d.bd == fx.d.bd) {
        fx.colbatch_matrix().noalias() += **xs[i] * xs[i+1]->colbatch_matrix();
      } else {
        DYNET_ASSERT(xs[i+1]->d.bd == 1 || xs[i+1]->d.bd == xs[i]->d.bd, "Failed dimension check in AffineTransform::forward");
        for(unsigned b = 0; b < fx.d.bd; ++b) {
          fx.batch_matrix(b).noalias() += xs[i]->batch_matrix(b) * xs[i+1]->batch_matrix(b);
        }
      }
    }
  }
}

template<class MyDevice>
void AffineTransform::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < xs.size(), "Failed boundary check in AffineTransform::backward");
  // Bias term
  if (i == 0) { // bias term
    size_t dx_size = dEdxi.d.size(), df_size = dEdf.d.size();
    if(dx_size == df_size) {
      dEdxi.tvec().device(*dev.edevice) += dEdf.tvec();
    } else {
      DYNET_ARG_CHECK(dEdxi.d.bd == 1, "In AffineTransform, broadcasting over columns with mini-batched inputs is not implemented yet");
      if(dEdxi.d[1] == dEdf.d[1]) {
        for(unsigned b = 0; b < dEdf.d.bd; ++b)
          (*dEdxi).noalias() += dEdf.batch_matrix(b);
      } else {
        Tensor mychip(dEdxi.d, dEdf.v, dEdf.device, dEdf.mem_pool);
        size_t len = dEdf.d.bd * dEdf.d[1];
        for(unsigned b = 0; b < len; ++b) {
          (*dEdxi).noalias() += *mychip;
          mychip.v += dx_size;
        }
      }
    }

  // Left argument of matrix multiply
  } else if (i % 2 == 1) {
    int max_b = max(dEdf.d.bd, xs[i+1]->d.bd);
    if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[i+1]->d.bd)) {
      (*dEdxi).noalias() += dEdf.colbatch_matrix() * xs[i+1]->colbatch_matrix().transpose();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += dEdf.batch_matrix(b) * xs[i+1]->batch_matrix(b).transpose();
    }
  } else {  // right argument of matrix multiply
    int max_b = max(xs[i-1]->d.bd, dEdf.d.bd);
    if(xs[i-1]->d.bd == 1 && dEdxi.d.bd == dEdf.d.bd) {
      dEdxi.colbatch_matrix().noalias() += (**xs[i-1]).transpose() * dEdf.colbatch_matrix();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += xs[i-1]->batch_matrix(b).transpose() * dEdf.batch_matrix(b);
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(AffineTransform)

}
