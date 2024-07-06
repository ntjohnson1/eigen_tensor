#pragma once
#include "utils/eigen_helpers.hpp"
#include <cmath>
#include <optional>
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
  // FIXME: Some ops are hard to validate without seeing the values
  // Some indexing is easier peeking at the values, otherwise will need to
  // use friend. Should think more carefully about this interface
  const std::vector<Eigen::Tensor<dtype, 2>> &factor_matrices() {
    return factor_matrices_;
  }
  const Eigen::Tensor<dtype, 1> &weights() { return weights_; }
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

  void normalize(size_t mode, int normtype = 2) {
    if (mode >= ndims()) {
      throw std::invalid_argument("Mode parameter is invalid; must be in the "
                                  "range of number of dimensions");
    }
    std::array<long, 2> offsets({0, 0});
    std::array<long, 2> extents({0, 1});
    for (long r = 0; r < ncomponents(); ++r) {
      offsets = {0, r};
      extents = {factor_matrices_[mode].dimension(0), 1};
      Eigen::Tensor<dtype, 2> slice =
          factor_matrices_[mode].slice(offsets, extents);
      // Look at if I can take an overloaded op as an argument to reduce the
      // need to make concrete first
      dtype tmp = p_norm(slice, normtype);
      if (tmp > 0.) {
        factor_matrices_[mode].slice(offsets, extents) =
            slice * dtype(1. / tmp);
      }
      weights_[r] *= tmp;
    }
  }

  void normalize(std::optional<size_t> weight_factor = std::nullopt,
                 bool sort = false, int normtype = 2) {
    normalize(weight_factor, sort, normtype, false);
  }

  void normalize(bool all_weights, bool sort = false, int normtype = 2) {
    normalize(std::nullopt, sort, normtype, all_weights);
  }

  // This is a little clunky stil, continue brainstorming options
  // I still like a builder method here, but that might just be from looking at
  // the rust
  void normalize(std::optional<size_t> weight_factor, bool sort = false,
                 int normtype = 2, bool all_weights = false) {
    for (size_t mode = 0; mode < ndims(); ++mode) {
      normalize(mode, normtype);
    }
    // Check that all weights are positive, flip sign of columns in first factor
    // matrix if negative weight found
    std::array<long, 2> extents({factor_matrices_[0].dimension(0), 1});
    for (long i = 0; i < ncomponents(); ++i) {
      if (weights_(i) < 0.) {
        weights_(i) *= -1.;
        std::array<long, 2> offsets({0, i});
        factor_matrices_[0].slice(offsets, extents) =
            factor_matrices_[0].slice(offsets, extents) * dtype(-1.);
      }
    }

    // Absorb weight into factors
    if (all_weights) {
      // All factors
      Eigen::Tensor<dtype, 2> d(weights_.size(), weights_.size());
      d.setZero();
      for (auto i = 0; i < weights_.size(); ++i) {
        d(i, i) = std::pow(weights_(i), dtype(1. / dtype(ndims())));
      }
      std::array<Eigen::IndexPair<int>, 1> dot_product = {
          Eigen::IndexPair<int>(1, 0)};
      for (auto i = 0; i < ndims(); ++i) {
        factor_matrices_[i] = factor_matrices_[i].contract(d, dot_product);
      }
      weights_.setConstant(1.);
    } else if (weight_factor.has_value()) {
      // Single factor
      auto idx = weight_factor.value();
      Eigen::Tensor<dtype, 2> d(weights_.size(), weights_.size());
      d.setZero();
      for (auto i = 0; i < weights_.size(); ++i) {
        d(i, i) = weights_(i);
      }
      std::array<Eigen::IndexPair<int>, 1> dot_product = {
          Eigen::IndexPair<int>(1, 0)};
      factor_matrices_[idx] = factor_matrices_[idx].contract(d, dot_product);
      weights_.setConstant(1.);
    }

    if (sort && ncomponents() > 1) {
      auto p = argsort(weights_);
      arrange(p);
    }
  }
};

} // namespace ttb
