#include "utils/khatrirao.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace ttb;

TEST_CASE("Smoke test", "[khatrirao]") {
  std::vector<Eigen::Tensor<float, 2>> factors;
  Eigen::Tensor<float, 2> original(2, 3);
  original.setZero();
  factors.push_back(original);
  auto result = khatrirao(factors);
}

TEST_CASE("Column Vectors", "[khatrirao]") {
  Eigen::Tensor<float, 2> factor(3, 1);
  factor.setConstant(2.);
  std::vector<Eigen::Tensor<float, 2>> factors = {factor, factor};
  auto result = khatrirao(factors);
  REQUIRE(result.dimension(0) == 9);
  Eigen::Tensor<bool, 2> correct = (result == result.constant(4.0f));
  Eigen::Tensor<int, 0> sum_correct = correct.cast<int>().sum();
  // Weird doing correct.all() yields false
  REQUIRE(sum_correct(0) == result.size());

  result = khatrirao(factors, true);
  REQUIRE(result.dimension(0) == 9);
  correct = (result == result.constant(4.0f));
  sum_correct = correct.cast<int>().sum();
  REQUIRE(sum_correct(0) == result.size());
};

TEST_CASE("Row Vectors", "[khatrirao]") {
  Eigen::Tensor<float, 2> factor(1, 3);
  factor.setConstant(2.);
  std::vector<Eigen::Tensor<float, 2>> factors = {factor, factor};
  auto result = khatrirao(factors);
  REQUIRE(result.dimension(0) == 1);
  REQUIRE(result.dimension(1) == 3);
  Eigen::Tensor<bool, 2> correct = (result == result.constant(4.0f));
  Eigen::Tensor<int, 0> sum_correct = correct.cast<int>().sum();
  // Weird doing correct.all() yields false
  REQUIRE(sum_correct(0) == result.size());

  result = khatrirao(factors, true);
  REQUIRE(result.dimension(0) == 1);
  REQUIRE(result.dimension(1) == 3);
  correct = (result == result.constant(4.0f));
  sum_correct = correct.cast<int>().sum();
  REQUIRE(sum_correct(0) == result.size());
};
