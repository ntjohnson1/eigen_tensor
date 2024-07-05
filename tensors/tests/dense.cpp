#include "tensors/dense.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Dense Constructor", "[dense]") {
  Eigen::Tensor<float, 3> original(1, 2, 3);
  ttb::Dense foo(original);
}

TEST_CASE("Dense ndims", "[dense]") {
  Eigen::Tensor<float, 3> original(1, 2, 3);
  ttb::Dense foo(original);
  REQUIRE(foo.ndims() == 3);
}

TEST_CASE("Dense norm", "[dense]") {
  Eigen::Tensor<float, 4> original(2, 2, 2, 2);
  original.setZero();
  ttb::Dense foo(original);
  REQUIRE(foo.norm() == 0.0);
  original.setConstant(1.);
  ttb::Dense bar(original);
  REQUIRE(bar.norm() == 4.0);
}

TEST_CASE("Dense innerprod", "[dense]") {
  Eigen::Tensor<float, 4> original(2, 2, 2, 2);
  original.setConstant(1.);
  ttb::Dense bar(original);
  REQUIRE(bar.innerprod(bar) == 16.0);
}

TEST_CASE("Dense mttkrp", "[dense]") {
  Eigen::Tensor<float, 3> array(2, 3, 4);
  array.setConstant(1.);
  ttb::Dense tensor(array);

  Eigen::Tensor<float, 2> factor_0(2, 2);
  factor_0.setValues({{1.0, 3.0}, {2.0, 4.0}});
  Eigen::Tensor<float, 2> factor_1(3, 2);
  factor_1.setValues({{5.0, 8.0}, {6.0, 9.0}, {7.0, 10.0}});
  Eigen::Tensor<float, 2> factor_2(4, 2);
  factor_2.setValues({{11.0, 15.0}, {12.0, 16.0}, {13.0, 17.0}, {14.0, 18.0}});
  std::vector<Eigen::Tensor<float, 2>> factors = {factor_0, factor_1, factor_2};

  Eigen::Tensor<float, 2> m0(2, 2);
  m0.setValues({{1800.0, 3564.0}, {1800.0, 3564.0}});
  m0 /= m0.constant(
      2.0); // Division is unpacking the weight from pyttb ktensor version

  Eigen::Tensor<float, 2> m1(3, 2);
  m1.setValues({{300.0, 924.0}, {300.0, 924.0}, {300.0, 924.0}});
  m1 /= m1.constant(2.0);

  Eigen::Tensor<float, 2> m2(4, 2);
  m2.setValues(
      {{108.0, 378.0}, {108.0, 378.0}, {108.0, 378.0}, {108.0, 378.0}});
  m2 /= m2.constant(2.0);

  auto result = tensor.mttkrp(factors, 0);
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      REQUIRE_THAT(result(i, j), Catch::Matchers::WithinRel(m0(i, j), 1e-8f));
    }
  }

  result = tensor.mttkrp(factors, 1);
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      REQUIRE_THAT(result(i, j), Catch::Matchers::WithinRel(m1(i, j), 1e-8f));
    }
  }

  result = tensor.mttkrp(factors, 2);
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      REQUIRE_THAT(result(i, j), Catch::Matchers::WithinRel(m2(i, j), 1e-8f));
    }
  }
}
