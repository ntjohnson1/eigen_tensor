// Eigen Tensor doesn't easily interop with Eigen Matrices
// so this povides some convenienvce features
#pragma once
#include <algorithm>
#include <array>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ttb {

// Lazier mimicking of Eigen lpNorm
template <typename dtype, int rank>
dtype p_norm(const Eigen::Tensor<dtype, rank> &tensor, int p) {
  Eigen::Tensor<dtype, 0> norm;
  if (p == 1) {
    norm = tensor.abs().sum();
  } else if (p == Eigen::Infinity) {
    norm = tensor.abs().maximum();
  } else {
    norm = tensor.abs().pow(p).sum().pow(dtype(1. / p));
  }
  return norm(0);
}

// Initially only support vector
template <typename dtype>
std::vector<size_t> argsort(const Eigen::Tensor<dtype, 1> &tensor) {
  std::vector<size_t> indices(tensor.size());
  std::sort(indices.begin(), indices.end(),
            [&tensor](size_t first, size_t second) {
              return tensor(first) < tensor(second);
            });
  return indices;
}

} // namespace ttb
