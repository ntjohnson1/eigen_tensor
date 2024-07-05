#pragma once
#include <array>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ttb {
// Could probably further constrain iterator to be explicit.
template <typename dtype, typename iterator>
Eigen::Tensor<dtype, 2> khatrirao(const iterator &begin, const iterator &end) {
  int ncol_first = (*begin).dimension(1);
  for (auto matrix = begin; matrix != end; ++matrix) {
    // Does Eigen tensor allow these to be statically asserted?
    if ((*matrix).dimension(1) != ncol_first) {
      throw std::invalid_argument(
          "All matrices must have the same number of columns");
    }
    if ((*matrix).NumDimensions != 2) {
      throw std::invalid_argument("Each factor must be a matrix");
    }
  }

  int p_remaining = (*begin).size() / ncol_first;
  Eigen::Tensor<dtype, 3> p(1, p_remaining, ncol_first);
  std::array<Eigen::Index, 3> offsets = {0, 0, 0};
  std::array<Eigen::Index, 3> extents = {1, p_remaining, ncol_first};
  p.slice(offsets, extents) = (*begin).reshape(extents);

  for (auto i = (begin + 1); i != end; ++i) {
    const int i_rows = (*i).dimension(0);
    Eigen::Tensor<dtype, 3> i_(i_rows, 1, ncol_first);
    i_.setZero();
    extents = {i_rows, 1, ncol_first};
    i_.slice(offsets, extents) = (*i).reshape(extents);
    extents = {1, p_remaining, ncol_first};
    p = p.reshape(extents);
    std::array<Eigen::Index, 3> i_bcast({1, p_remaining, 1});
    std::array<Eigen::Index, 3> p_bcast({i_rows, 1, 1});
    // I tried to do this as a one liner and even with eval calls it wouldn't
    // resolve. Not sure if this hits our efficiency at all
    Eigen::Tensor<dtype, 3> tmp_p = p.broadcast(p_bcast);
    p = i_.broadcast(i_bcast) * tmp_p;
    p_remaining = p.size() / ncol_first;
  }
  p_remaining = p.size() / ncol_first;
  std::array<Eigen::Index, 2> final_shape = {p_remaining, ncol_first};
  // TODO check if this is right, we do some weird ROW/COL permutations in
  // python and rust
  return p.reshape(final_shape);
}

template <typename dtype>
Eigen::Tensor<dtype, 2>
khatrirao(const std::vector<Eigen::Tensor<dtype, 2>> &factors,
          bool reverse = false) {
  if (reverse) {
    return khatrirao<dtype>(factors.rbegin(), factors.rend());
  }
  return khatrirao<dtype>(factors.begin(), factors.end());
};
} // namespace ttb
