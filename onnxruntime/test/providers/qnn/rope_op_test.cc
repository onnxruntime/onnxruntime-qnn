// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

constexpr float kDefaultRopeOpToleranceFp32 = 1e-4f;

template <typename DataType = float>
static void RunRopeOpTest(const TestInputDef<DataType>& input_def,
                          const TestInputDef<int64_t>& position_ids_def,
                          const TestInputDef<DataType>& cos_cache_def,
                          const TestInputDef<DataType>& sin_cache_def,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          int opset_version,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          float fp32_abs_err = kDefaultRopeOpToleranceFp32) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>("RotaryEmbedding",
                                                     {input_def},
                                                     {position_ids_def},
                                                     {cos_cache_def, sin_cache_def},
                                                     attrs,
                                                     kMSDomain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

TEST_F(QnnHTPBackendTests, RotaryEmbedding_4D_NonInterleaved_FullRotation) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RotaryEmbedding_4D_Interleaved) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(1)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// Partial rotation: rotary_dim < head_size, preserving some dimensions without positional encoding
TEST_F(QnnHTPBackendTests, RotaryEmbedding_4D_PartialRotation) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = 4;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// 3D input requires num_heads attribute to split the hidden dimension
TEST_F(QnnHTPBackendTests, RotaryEmbedding_3D_NonInterleaved) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t hidden_size = num_heads * head_size;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * hidden_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, seq_len, hidden_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RotaryEmbedding_3D_Interleaved) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t hidden_size = num_heads * head_size;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * hidden_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, seq_len, hidden_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(1)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// Position IDs enable non-contiguous position patterns (e.g., for sparse attention or cached KV)
TEST_F(QnnHTPBackendTests, RotaryEmbedding_WithPositionIds) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3, 0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// 3D caches allow batch-specific position encodings
TEST_F(QnnHTPBackendTests, RotaryEmbedding_4D_With3DCaches) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3, 0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({batch_size, seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({batch_size, seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// Test default behavior when rotary_embedding_dim is not specified (defaults to full head_size)
TEST_F(QnnHTPBackendTests, RotaryEmbedding_DefaultRotaryDim) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (head_size / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (head_size / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, head_size / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, head_size / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RotaryEmbedding_LargerDimensions) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t num_heads = 4;
  constexpr int64_t seq_len = 8;
  constexpr int64_t head_size = 16;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids(batch_size * seq_len);
  for (int64_t i = 0; i < batch_size * seq_len; ++i) {
    position_ids[i] = i % seq_len;
  }

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// Edge case: seq_len=1 is common during autoregressive generation
TEST_F(QnnHTPBackendTests, RotaryEmbedding_SingleSequenceLength) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 1;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

// Edge case: odd rotary_dim value (rotary_dim/2 uses integer division)
TEST_F(QnnHTPBackendTests, RotaryEmbedding_OddRotaryDim) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = 5;

  std::vector<float> input_data = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
                       TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
                       TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
                       {utils::MakeAttribute("interleaved", static_cast<int64_t>(0)),
                        utils::MakeAttribute("num_heads", num_heads),
                        utils::MakeAttribute("rotary_embedding_dim", rotary_dim)},
                       1,
                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
