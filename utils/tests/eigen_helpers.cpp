#include "utils/eigen_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace ttb;

TEST_CASE("P Norm", "[eigen_helpers]") {
  Eigen::Tensor<float, 2> tensor(2, 2);
  tensor.setConstant(1.);
  auto norm_2 = p_norm(tensor, 2);
  REQUIRE(norm_2 == 2.0f);
  auto norm_1 = p_norm(tensor, 1);
  REQUIRE(norm_1 == 4.0f);
  auto norm_inf = p_norm(tensor, Eigen::Infinity);
  REQUIRE(norm_inf == 1.0f);
}
