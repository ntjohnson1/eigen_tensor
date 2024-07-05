#include "tensors/kruskal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace ttb;

TEST_CASE("Kruskal Constructor", "[kruskal]") {
  Eigen::Tensor<float, 2> factor(2, 2);
  std::vector<Eigen::Tensor<float, 2>> factors{factor, factor};
  Eigen::Tensor<float, 1> weights(2);
  Kruskal foo(factors);
  Kruskal bar(factors, weights);
}

TEST_CASE("Kruskal ndims", "[kruskal]") {
  Eigen::Tensor<float, 2> factor_0(2, 2);
  Eigen::Tensor<float, 2> factor_1(3, 2);
  std::vector<Eigen::Tensor<float, 2>> factors{factor_0, factor_1};
  Eigen::Tensor<float, 1> weights(2);
  Kruskal tensor(factors, weights);
  REQUIRE(tensor.ndims() == factors.size());
}

TEST_CASE("Kruskal ncomponents", "[kruskal]") {
  Eigen::Tensor<float, 2> factor_0(2, 2);
  Eigen::Tensor<float, 2> factor_1(3, 2);
  std::vector<Eigen::Tensor<float, 2>> factors{factor_0, factor_1};
  Eigen::Tensor<float, 1> weights(2);
  Kruskal tensor(factors, weights);
  REQUIRE(tensor.ncomponents() == weights.size());
}

TEST_CASE("Kruskal norm", "[kruskal]") {
  Eigen::Tensor<float, 2> factor_0(2, 2);
  Eigen::Tensor<float, 2> factor_1(3, 2);
  factor_0.setConstant(0.);
  factor_1.setConstant(0.);
  std::vector<Eigen::Tensor<float, 2>> factors{factor_0, factor_1};
  Kruskal tensor_0(factors);
  REQUIRE_THAT(tensor_0.norm(), Catch::Matchers::WithinRel(0.0f, 1e-8f));

  factor_0.setConstant(1.);
  factors = {factor_0, factor_0};
  Kruskal tensor_1(factors);
  REQUIRE_THAT(tensor_1.norm(), Catch::Matchers::WithinRel(4.0f, 1e-8f));
}

TEST_CASE("Kruskal isequal", "[kruskal]") {
  Eigen::Tensor<float, 1> weights(2);
  weights.setValues({1., 2.});
  Eigen::Tensor<float, 2> factor_0(2, 2);
  factor_0.setValues({{1., 2.}, {3., 4.}});
  Eigen::Tensor<float, 2> factor_1(2, 2);
  factor_1.setValues({{5., 6.}, {7., 8.}});
  std::vector<Eigen::Tensor<float, 2>> factors{factor_0, factor_1};
  Kruskal tensor_0(factors, weights);
  Kruskal tensor_1(factors, weights);
  REQUIRE(tensor_0.isequal(tensor_1));

  // mismatch weights
  weights(0) = 0.;
  Kruskal tensor_2(factors, weights);
  REQUIRE(!tensor_0.isequal(tensor_2));

  // mistmatch factors
  weights(0) = 1.;
  factors[0](0, 0) = 0.;
  Kruskal tensor_3(factors, weights);
  REQUIRE(!tensor_0.isequal(tensor_3));
}

TEST_CASE("Kruskal arrange", "[kruskal]") {
  Eigen::Tensor<float, 1> weights(2);
  weights.setValues({1., 2.});
  Eigen::Tensor<float, 2> factor_0(2, 2);
  factor_0.setValues({{1., 2.}, {3., 4.}});
  Eigen::Tensor<float, 2> factor_1(2, 2);
  factor_1.setValues({{5., 6.}, {7., 8.}});
  std::vector<Eigen::Tensor<float, 2>> factors{factor_0, factor_1};
  Kruskal tensor_0(factors, weights);
  Kruskal tensor_1(factors, weights);
  tensor_1.arrange({1, 0});
  tensor_1.arrange({1, 0});
  REQUIRE(tensor_0.isequal(tensor_1));
}
