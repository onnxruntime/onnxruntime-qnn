// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for the shape/permutation helpers in
// qnn_node_group/utils.h. Both helpers are defined inline in the header, so
// these tests run in every build (no QNN_EP_INTERNAL_SYMBOL_ACCESS guard).

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace test {

// =============================================================================
// qnn::IsReshapePermutable
// =============================================================================

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_SameShapeIsPermutable) {
  EXPECT_TRUE(qnn::IsReshapePermutable({2, 3, 4}, {2, 3, 4}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_ScalarSameRank) {
  EXPECT_TRUE(qnn::IsReshapePermutable({}, {}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_RankMismatchRejected) {
  EXPECT_FALSE(qnn::IsReshapePermutable({1, 4, 4}, {1, 16}));
  EXPECT_FALSE(qnn::IsReshapePermutable({1, 16}, {1, 4, 4}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_NegativeDimRejected) {
  EXPECT_FALSE(qnn::IsReshapePermutable({-1, 4}, {4, -1}));
  EXPECT_FALSE(qnn::IsReshapePermutable({2, 4}, {-1, 4}));
  EXPECT_FALSE(qnn::IsReshapePermutable({-1, 4}, {2, 4}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_NonOneDimsDifferByValue) {
  // {2,3,4} and {2,4,3} have the same set of non-1 dims but a different sequence.
  // The sequences must be identical, not just multisets — order-preserving.
  EXPECT_FALSE(qnn::IsReshapePermutable({2, 3, 4}, {2, 4, 3}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_NonOneDimsDifferByCount) {
  // Extra 5 in the output has no counterpart in the input.
  EXPECT_FALSE(qnn::IsReshapePermutable({2, 3, 4}, {2, 3, 5}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_SizeOneDimsCanMoveFreely) {
  EXPECT_TRUE(qnn::IsReshapePermutable({3, 4, 1, 1}, {3, 1, 4, 1}));
  EXPECT_TRUE(qnn::IsReshapePermutable({1, 4, 4, 1}, {1, 1, 4, 4}));
  EXPECT_TRUE(qnn::IsReshapePermutable({1, 2, 1, 3}, {2, 1, 1, 3}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_AllOnesAllRanks) {
  EXPECT_TRUE(qnn::IsReshapePermutable({1, 1, 1}, {1, 1, 1}));
  EXPECT_TRUE(qnn::IsReshapePermutable({1}, {1}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_ZeroDimIsNotNegative) {
  // Zero-size dims are non-1 and must appear in the same relative order as any
  // other non-1 dim -- reordering them is not a legal Reshape-as-Transpose.
  EXPECT_TRUE(qnn::IsReshapePermutable({0, 3}, {0, 3}));
  EXPECT_FALSE(qnn::IsReshapePermutable({0, 3}, {3, 0}));
  EXPECT_FALSE(qnn::IsReshapePermutable({0, 3}, {0, 4}));
}

TEST(QnnUnit_NodeGroupUtilsTest, IsReshapePermutable_EqualNonUnitDimsAtDifferentPositions) {
  // Two non-1 dims of equal size at non-adjacent positions — still permutable
  // because the non-1 sub-sequence [2, 2] matches on both sides.
  EXPECT_TRUE(qnn::IsReshapePermutable({2, 1, 2, 1}, {2, 2, 1, 1}));
  EXPECT_TRUE(qnn::IsReshapePermutable({2, 1, 2, 1}, {1, 1, 2, 2}));
}

// =============================================================================
// qnn::ComputeReshapePerm
//
// Contract: output_shape[i] == input_shape[perm[i]]. Non-1 dims are matched
// left-to-right (preserving their relative order); size-1 dims greedily take the
// next unused slot.
// =============================================================================

// Helper: reconstruct the output shape from perm and check it matches expected.
static bool ApplyPerm(const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& perm,
                      const std::vector<int64_t>& expected_output) {
  if (perm.size() != expected_output.size()) return false;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] < 0 || static_cast<size_t>(perm[i]) >= input_shape.size()) return false;
    if (input_shape[static_cast<size_t>(perm[i])] != expected_output[i]) return false;
  }
  return true;
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_IdentityShapeIdentityPerm) {
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({2, 3, 4}, {2, 3, 4}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 1, 2}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_ScalarIsEmptyPerm) {
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({}, {}, perm);
  EXPECT_TRUE(perm.empty());
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_SwapSizeOneDim) {
  // [3, 4, 1, 1] -> [3, 1, 4, 1]: 4 moves from index 1 to index 2. Size-1 dims
  // greedily take the next unused slot: perm[1] picks input[2] (the first '1'),
  // perm[3] picks input[3] (the remaining '1').
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({3, 4, 1, 1}, {3, 1, 4, 1}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 2, 1, 3}));
  EXPECT_TRUE(ApplyPerm({3, 4, 1, 1}, perm, {3, 1, 4, 1}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_MoveNonUnitDimAcrossOnes) {
  // [1, 4, 4, 1] -> [1, 1, 4, 4]: first output '1' picks input[0], second output
  // '1' picks input[3], then the two 4s pick input[1], input[2].
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({1, 4, 4, 1}, {1, 1, 4, 4}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 3, 1, 2}));
  EXPECT_TRUE(ApplyPerm({1, 4, 4, 1}, perm, {1, 1, 4, 4}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_EqualNonUnitDimsPreserveOrder) {
  // [2, 1, 2, 1] -> [2, 2, 1, 1]: the two output '2's must map to input[0] and
  // input[2] in that order (relative order of non-1 dims must be preserved). If
  // the greedy walk instead picked input[2] for the first output '2', the
  // resulting perm would be [2, 0, 1, 3] and represent a genuinely different
  // data mapping.
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({2, 1, 2, 1}, {2, 2, 1, 1}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 2, 1, 3}));
  EXPECT_TRUE(ApplyPerm({2, 1, 2, 1}, perm, {2, 2, 1, 1}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_AllOnes) {
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({1, 1, 1}, {1, 1, 1}, perm);
  // Greedy: each output '1' picks the next unused input '1' -> identity perm.
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 1, 2}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_ZeroSizeDim) {
  // Zero-size dims are treated as any other non-1 dim -- matched left-to-right.
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({0, 3}, {0, 3}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(ApplyPerm({0, 3}, perm, {0, 3}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_Rank5MixedOnes) {
  // [1, 2, 1, 3, 1] -> [2, 1, 3, 1, 1]. Non-1 order preserved: 2 goes first,
  // then 3. Size-1 dims take the next unused input '1' slots left-to-right.
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({1, 2, 1, 3, 1}, {2, 1, 3, 1, 1}, perm);
  EXPECT_EQ(perm, (std::vector<int64_t>{1, 0, 3, 2, 4}));
  EXPECT_TRUE(ApplyPerm({1, 2, 1, 3, 1}, perm, {2, 1, 3, 1, 1}));
}

TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_ReversedNonUnitDims) {
  // Distinct non-1 dims in reversed order across size-1 dims.
  // [1, 5, 1, 7] -> [7, 1, 5, 1]. Non-1 output order 7,5 differs from input
  // order 5,7 -- IsReshapePermutable would reject this, so ComputeReshapePerm
  // makes no promise. Verify we do not crash and produce a rank-matching perm.
  std::vector<int64_t> perm;
  qnn::ComputeReshapePerm({1, 5, 1, 7}, {7, 1, 5, 1}, perm);
  EXPECT_EQ(perm.size(), 4u);
}

// A perm returned for a permutable shape must be a valid permutation of
// [0..rank-1] with no duplicates. Sweep across a handful of shapes.
TEST(QnnUnit_NodeGroupUtilsTest, ComputeReshapePerm_ProducesValidPermutation) {
  struct Case {
    std::vector<int64_t> in;
    std::vector<int64_t> out;
  };
  const Case cases[] = {
      {{3, 4, 1, 1}, {3, 1, 4, 1}},
      {{1, 4, 4, 1}, {1, 1, 4, 4}},
      {{2, 1, 2, 1}, {2, 2, 1, 1}},
      {{1, 2, 1, 3, 1}, {2, 1, 3, 1, 1}},
      {{2, 3, 4}, {2, 3, 4}},
      {{1, 1, 1}, {1, 1, 1}},
  };
  for (const auto& c : cases) {
    ASSERT_TRUE(qnn::IsReshapePermutable(c.in, c.out))
        << "precondition failed for case with in.size()=" << c.in.size();
    std::vector<int64_t> perm;
    qnn::ComputeReshapePerm(c.in, c.out, perm);
    ASSERT_EQ(perm.size(), c.in.size());
    std::vector<bool> seen(perm.size(), false);
    for (int64_t p : perm) {
      ASSERT_GE(p, 0);
      ASSERT_LT(static_cast<size_t>(p), perm.size());
      ASSERT_FALSE(seen[static_cast<size_t>(p)]) << "duplicate index in perm";
      seen[static_cast<size_t>(p)] = true;
    }
    EXPECT_TRUE(ApplyPerm(c.in, perm, c.out));
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
