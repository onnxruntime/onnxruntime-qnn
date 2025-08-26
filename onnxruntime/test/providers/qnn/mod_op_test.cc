// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Mod operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunModTest(const std::vector<TestInputDef<DataType>>& input_defs,
                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       const std::string& backend_name = "cpu",
                       int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<DataType>("Mod", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Mod with dynamic divisor.
TEST_F(QnnCPUBackendTests, Mod_dynamic_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int32_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int32_t>(divisor_shape, 1.0f, 10.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, false, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int32_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, false, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);
}

// On linux as an example.
// For positive test, at index #1 -96.3294 mod  1.04149: expects -0.51226 but gets 0.529228.
// For negative test, at index #2  78.8961 mod -5.66191: expects  5.29132795 but gets -0.370574951.
// Both expected values are wrong, correct result should follow onnx doc:
//    "If the fmod attribute is set to 0, T is constrained to integer data types and the semantics follow that of the Python %-operator.
//     The sign of the result is that of the divisor."
//  Where the result from QNN EP is correct (same sign and same value from python.)
// Therefore, this test is disabled.
TEST_F(QnnCPUBackendTests, DISABLED_Fmod_dynamic_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<float>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1};
  auto divisor = rand_gen_.Uniform<float>(divisor_shape, 1.0f, 10.0f);
  RunModTest<float>({TestInputDef<float>({1, 4, 5}, false, dividend),
                     TestInputDef<float>({1}, false, divisor)},
                    {utils::MakeAttribute("fmod", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All);

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<float>(divisor_shape, -10.0f, -1.0f);
  std::cout << std::endl;
  RunModTest<float>({TestInputDef<float>({1, 4, 5}, false, dividend),
                     TestInputDef<float>({1}, false, neg_divisor)},
                    {utils::MakeAttribute("fmod", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All);
}

// Test that Mod with static divisor.
TEST_F(QnnCPUBackendTests, Mod_static_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int32_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int32_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, true, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int32_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, true, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);
}
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Mod with dynamic divisor.
TEST_F(QnnHTPBackendTests, Mod_dynamic_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int64_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int64_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, false, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int64_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, false, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");
}

// Test that Mod with static divisor.
TEST_F(QnnHTPBackendTests, Mod_static_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int64_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int64_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, true, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int64_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, true, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");
}

// Fmod on HTP is not supported. QAIRT 3.37 Failed to finalize QNN graph 1002.
TEST_F(QnnHTPBackendTests, Fmod_dynamic_Divisor_Unsupported) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<float>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<float>(divisor_shape, 1.0f, 10.0f);
  RunModTest<float>({TestInputDef<float>({1, 4, 5}, false, dividend),
                     TestInputDef<float>({1, 5}, false, divisor)},
                    {utils::MakeAttribute("fmod", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::None,
                    "htp");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
