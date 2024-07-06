#include "tensors/kruskal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

using namespace ttb;

// TODO make test helpers
template <typename dtype>
void check_matrix(Eigen::Tensor<dtype, 2> first, Eigen::Tensor<dtype, 2> second,
                  dtype thresh = 1e-8) {
  REQUIRE(first.dimension(0) == second.dimension(0));
  REQUIRE(first.dimension(1) == second.dimension(1));
  for (auto i = 0; i < first.dimension(0); ++i) {
    for (auto j = 0; j < first.dimension(1); ++j) {
      REQUIRE_THAT(first(i, j),
                   Catch::Matchers::WithinRel(second(i, j), thresh));
    }
  }
}

// TODO generalize
template <typename dtype>
void check_vector(Eigen::Tensor<dtype, 1> first, Eigen::Tensor<dtype, 1> second,
                  dtype thresh = 1e-8) {
  REQUIRE(first.dimension(0) == second.dimension(0));
  for (auto i = 0; i < first.dimension(0); ++i) {
    REQUIRE_THAT(first(i), Catch::Matchers::WithinRel(second(i), thresh));
  }
}

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

TEST_CASE("Kruskal normalize", "[kruskal]") {
  // Values from pyttb from MATLAB would be nice to figure out a first
  // principles approach
  Eigen::Tensor<double, 1> weights(2);
  weights.setValues({2., 2.});
  Eigen::Tensor<double, 2> factor_0(2, 2);
  factor_0.setValues({{1., 3.}, {2., 4.}});
  Eigen::Tensor<double, 2> factor_1(3, 2);
  factor_1.setValues({{5., 8.}, {6., 9.}, {7., 10.}});
  Eigen::Tensor<double, 2> factor_2(4, 2);
  factor_2.setValues({{11.0, 15.0}, {12.0, 16.0}, {13.0, 17.0}, {14.0, 18.0}});
  std::vector<Eigen::Tensor<double, 2>> factors{factor_0, factor_1, factor_2};
  Kruskal tensor_0(factors, weights);
  Kruskal tensor_1(factors, weights);

  SECTION("Normalize mode 1") {
    weights.setValues({20.97617696340303, 31.304951684997057});
    factor_1.setValues({{0.4767312946227962, 0.5111012519999519},
                        {0.5720775535473555, 0.5749889084999459},
                        {0.6674238124719146, 0.6388765649999399}});
    tensor_0.normalize(size_t(1));
    check_vector(tensor_0.weights(), weights);
    check_matrix(tensor_0.factor_matrices()[1], factor_1);
  }
  SECTION("Normalize with defaults") {
    weights.setValues({1177.285012220915, 5177.161384388167});
    factor_0.setValues({{0.4472135954999579, 0.6}, {0.8944271909999159, 0.8}});
    factor_1.setValues({{0.4767312946227962, 0.5111012519999519},
                        {0.5720775535473555, 0.5749889084999459},
                        {0.6674238124719146, 0.6388765649999399}});
    factor_2.setValues({{0.4382504900892777, 0.4535055413676754},
                        {0.4780914437337575, 0.4837392441255204},
                        {0.5179323973782373, 0.5139729468833655},
                        {0.5577733510227171, 0.5442066496412105}});
    tensor_0.normalize();
    check_vector(tensor_0.weights(), weights);
    check_matrix(tensor_0.factor_matrices()[0], factor_0);
    check_matrix(tensor_0.factor_matrices()[1], factor_1);
    check_matrix(tensor_0.factor_matrices()[2], factor_2);
  }
  // TODO negative weights, 1-norm, weight factor 1, normalize and sort,
  // all_weights
}
