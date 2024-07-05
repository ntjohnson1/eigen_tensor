#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

namespace ttb {

template <typename dtype> class Kruskal {
protected:
  std::vector<Eigen::Tensor<dtype, 2>> factor_matrices_;
  Eigen::Tensor<dtype, 1> weights_;

public:
  Kruskal(std::vector<Eigen::Tensor<dtype, 2>> factor_matrices,
          Eigen::Tensor<dtype, 1> weights)
      : factor_matrices_(factor_matrices) {
    weights_ = weights;
    for (const auto &factor : factor_matrices_) {
      if (weights_.size() != factor.dimension(1)) {
        throw std::invalid_argument(
            "Size of factor matrix does not match number of weights");
      }
    }
  }
  Kruskal(std::vector<Eigen::Tensor<dtype, 2>> factor_matrices)
      : factor_matrices_(factor_matrices) {
    Eigen::Tensor<dtype, 1> default_weights(factor_matrices_[0].dimension(1));
    default_weights.setConstant(1.);
    weights_ = default_weights;
  }
  size_t ndims() const { return size_t(factor_matrices_.size()); }
  size_t ncomponents() const { return size_t(weights_.size()); }
  dtype norm() const {
    std::array<size_t, 2> row_shape({ncomponents(), 1});
    std::array<size_t, 2> col_shape({1, ncomponents()});
    std::array<Eigen::IndexPair<int>, 1> dot_product = {
        Eigen::IndexPair<int>(1, 0)};
    auto row = weights_.reshape(row_shape);
    auto col = weights_.reshape(col_shape);
    Eigen::Tensor<dtype, 2> coef_matrix = row.contract(col, dot_product);

    std::array<size_t, 2> transpose({1, 0});
    for (const auto &factor : factor_matrices_) {
      auto fTf = factor.shuffle(transpose).contract(factor, dot_product);
      coef_matrix *= fTf;
    }
    Eigen::Tensor<dtype, 0> two_norm = coef_matrix.sum().abs().sqrt();
    return two_norm(0);
  }
  bool isequal(const Kruskal<dtype> other) const {
    if (ncomponents() != other.ncomponents() || ndims() != other.ndims()) {
      return false;
    }
    for (auto i = 0; i < weights_.size(); ++i) {
      if (weights_(i) != other.weights_(i)) {
        return false;
      }
    }
    // Apparently eigen tensor `all` reduction doesn't short circuit so
    // unrolling here
    for (auto i = 0; i < ndims(); ++i) {
      for (auto j = 0; j < factor_matrices_[i].dimension(0); ++j) {
        for (auto k = 0; k < factor_matrices_[i].dimension(1); ++k) {
          if (factor_matrices_[i](j, k) != other.factor_matrices_[i](j, k)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  void arrange(const std::vector<size_t> &permutation) {
    // Only a partial implementation of arrange to get normalize settled
    if (permutation.size() == ncomponents()) {
      Eigen::Tensor<dtype, 1> swap = weights_.constant(0.);
      for (auto i = 0; i < weights_.size(); ++i) {
        swap(i) = weights_(permutation[i]);
      }
      weights_ = swap;
      for (size_t i = 0; i < ndims(); ++i) {
        Eigen::Tensor<dtype, 2> swap_f = factor_matrices_[i].constant(0.);
        std::array<long, 2> extents({swap_f.dimension(0), 1});
        for (size_t j = 0; j < swap_f.dimension(1); ++j) {
          std::array<size_t, 2> offset({0, j});
          std::array<size_t, 2> offset_f({0, permutation[j]});
          swap_f.slice(offset, extents) =
              factor_matrices_[i].slice(offset_f, extents);
        }
        factor_matrices_[i] = swap_f;
      }
      return;
    }
    throw std::invalid_argument(
        "Permutations not equal to number of components aren't supported yet");
  }
};

} // namespace ttb
