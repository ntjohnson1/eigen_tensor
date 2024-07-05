#pragma once
#include "utils/khatrirao.hpp"
#include <array>
#include <functional>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ttb {

template <typename dtype, int rank> class Dense {
protected:
  Eigen::Tensor<dtype, rank> tensor_;

public:
  Dense(Eigen::Tensor<dtype, rank> tensor) : tensor_(tensor) {}

  size_t ndims() const { return size_t(tensor_.NumDimensions); }

  dtype norm() const {
    Eigen::Tensor<dtype, 0> two_norm = tensor_.square().sum().sqrt();
    return two_norm(0);
  }

  dtype innerprod(Dense<dtype, rank> &other) const {
    std::array<Eigen::IndexPair<int>, rank> contraction_dims;
    for (int i = 0; i < rank; ++i) {
      contraction_dims[i] = Eigen::IndexPair<int>(i, i);
    }
    Eigen::Tensor<dtype, 0> result =
        tensor_.contract(other.tensor_, contraction_dims);
    return result(0);
  }

  Eigen::Tensor<dtype, 2>
  mttkrp(const std::vector<Eigen::Tensor<dtype, 2>> &factors, size_t n) const {
    static_assert(rank >= 2,
                  "MTTKRP is invalid for tensors wiht fewer than 2 dimensions");
    if (factors.size() != rank) {
      throw std::invalid_argument(
          "Must provide as many factors as tensor rank");
    }

    const size_t range =
        (n == 0) ? factors[1].dimension(1) : factors[0].dimension(1);

    // Check dimensions match
    for (auto i = 0; i < factors.size(); ++i) {
      if (i == n)
        continue;
      // TODO decide how to expose shape from eigen tensor
      if (factors[i].dimension(0) != tensor_.dimension(i)) {
        throw std::invalid_argument("Entry " + std::to_string(i) +
                                    " of list of arrays is the wrong size");
      }
    }

    auto dims = tensor_.dimensions();
    size_t szl =
        std::accumulate(dims.begin(), dims.begin() + n, 1, std::multiplies<>{});
    size_t szr = std::accumulate(dims.begin() + n + 1, dims.end(), 1,
                                 std::multiplies<>{});
    size_t szn = tensor_.dimension(n);

    // FIXME remove after development
    if (szl * szr * szn != (size_t)tensor_.size()) {
      throw std::runtime_error(
          "Your math is bad the dimensions don't have proper coverage");
    }

    if (n == 0) {
      // factors[1:rank] reversed
      auto ur = khatrirao<dtype>(factors.rbegin(), factors.rend() - 1);
      std::array<size_t, 2> shape({szn, szr});
      auto y = tensor_.reshape(shape);
      std::array<Eigen::IndexPair<int>, 1> dot_product = {
          Eigen::IndexPair<int>(1, 0)};
      return y.contract(ur, dot_product);
    } else if (n == (rank - 1)) {
      // factors[0:rank-1] reversed
      auto ul = khatrirao<dtype>(factors.rbegin() + 1, factors.rend());
      std::array<size_t, 2> shape({szl, szn});
      std::array<size_t, 2> transpose({1, 0});
      auto y = tensor_.reshape(shape);
      std::array<Eigen::IndexPair<int>, 1> dot_product = {
          Eigen::IndexPair<int>(1, 0)};
      return y.shuffle(transpose).contract(ul, dot_product);
    }
    // factors[n+1, rank] reversed
    auto ul = khatrirao<dtype>(factors.rbegin(), factors.rend() - n - 1);
    const std::array<size_t, 3> ur_shape({szl, 1, range});
    // factors[0, n] reversed then reshaped
    const Eigen::Tensor<dtype, 3> ur =
        khatrirao<dtype>(factors.rbegin() + (rank - n), factors.rend())
            .reshape(ur_shape);
    const auto y_leftover = tensor_.size() / szr;
    std::array<size_t, 2> shape_2d({y_leftover, szr});
    auto y = tensor_.reshape(shape_2d);
    std::array<Eigen::IndexPair<int>, 1> dot_product = {
        Eigen::IndexPair<int>(1, 0)};
    auto y_ul = y.contract(ul, dot_product);
    std::array<size_t, 3> shape_3d({szl, szn, range});
    Eigen::Tensor<dtype, 3> y_3d = y_ul.reshape(shape_3d);
    Eigen::Tensor<dtype, 2> v(szn, range);
    v.setConstant(1.);
    std::array<size_t, 2> transpose({1, 0});
    std::array<size_t, 2> y_shape({szl, szn});
    std::array<size_t, 2> ur_slice({szl, 1});
    for (size_t r = 0; r < range; ++r) {
      std::array<size_t, 2> offsets_2d = {0, r};
      std::array<size_t, 2> extents_2d = {szn, 1};
      std::array<size_t, 3> offsets_3d = {0, 0, r};
      std::array<size_t, 3> extents_3d = {szl, szn, 1};
      std::array<size_t, 2> ur_bcast = {1, szn};
      // y_3d[:,:,r] -> 2d [szl, szn] -> transpose
      Eigen::Tensor<dtype, 2> prepare_y = y_3d.slice(offsets_3d, extents_3d)
                                              .reshape(y_shape)
                                              .shuffle(transpose);
      extents_3d = {szl, 1, 1};
      // ur[:,:,r] -> 2d [szl, 1]
      Eigen::Tensor<dtype, 2> prepare_ur =
          // auto prepare_ur =
          ur.slice(offsets_3d, extents_3d).reshape(ur_slice);
      v.slice(offsets_2d, extents_2d) =
          prepare_y.contract(prepare_ur, dot_product);
    }

    return v;
  }
};

} // namespace ttb
