/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_fallback.cpp

Abstract:

    This module implements fallback (non-AVX2) quantized n-bit GEMM kernels.

--*/

#include <algorithm>
#include <cassert>
#include <cstring> // for memcpy
#include <cmath>   // for fabsf

#include "qnbitgemm.h"

static inline int nearest_int(float fval)
{
    // Clamp large values for safety
    assert(fabsf(fval) <= 4194303.f);
    return static_cast<int>(std::round(fval));
}

void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }

        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = static_cast<int8_t>(std::min(127, v));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = static_cast<int16_t>(sum);
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

void dequantize_row_q8_K(const block_q8_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void QuantizeARow_Q8_K(
    size_t /*BlkLen*/,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    block_q8_K* y = reinterpret_cast<block_q8_K*>(QuantA);
    quantize_row_q8_K_ref(A, y, CountK);
}

void Quantize_Q8_K(
    size_t BlkLen,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    std::byte* QuantA
)
{
    const float* ARowPtr = A;
    std::byte* QuantARowPtr = static_cast<std::byte*>(QuantA);
    size_t QuantAStride = ((K + BlkLen - 1) / BlkLen) * sizeof(block_q8_K);

    for (size_t m = 0; m < M; ++m) {
        QuantizeARow_Q8_K(BlkLen, ARowPtr, K, QuantARowPtr);
        ARowPtr += lda;
        QuantARowPtr += QuantAStride;
    }
}

void DequantizeARow_Q8_K(
    size_t /*BlkLen*/,
    float* A,
    size_t CountK,
    const std::byte* QuantA
)
{
    const block_q8_K* x = reinterpret_cast<const block_q8_K*>(QuantA);
    dequantize_row_q8_K(x, A, CountK);
}

void Dequantize_Q8_K(
    size_t BlkLen,
    float* A,
    size_t M,
    size_t K,
    size_t lda,
    const std::byte* QuantA
)
{
    float* ARowPtr = A;
    const std::byte* QuantARowPtr = static_cast<const std::byte*>(QuantA);
    size_t QuantAStride = ((K + BlkLen - 1) / BlkLen) * sizeof(block_q8_K);

    for (size_t m = 0; m < M; ++m) {
        DequantizeARow_Q8_K(BlkLen, ARowPtr, K, QuantARowPtr);
        ARowPtr += lda;
        QuantARowPtr += QuantAStride;
    }
}

// Workspace size for fallback
size_t QTernaryBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t /*N*/,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    assert(ComputeType == SQNBIT_CompInt8);
    assert(BlkLen == QK_K);
    (void)BlkLen;
    (void)ComputeType;

    const size_t BlockCountK = (K + QK_K - 1) / QK_K;
    return M * BlockCountK * sizeof(block_q8_K);
}

// QuantB pack size
size_t QTernaryBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    assert(ComputeType == SQNBIT_CompInt8);
    assert(BlkLen == QK_K);
    (void)BlkLen;
    (void)ComputeType;

    const size_t BlockCountK = (K + QK_K - 1) / QK_K;
    return BlockCountK * N * sizeof(block_tq1_0);
}

size_t SQTernaryBitGemmKernel_TQ1_0_Q8_K(
    size_t /*BlkLen*/,
    const std::byte* QuantA,
    const std::byte* QuantB,
    const float* /*QuantBScale*/,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t /*BlockCountK*/,
    size_t ldc,
    const float* Bias
)
{

    const size_t BlkCountK = (CountK + QK_K - 1) / QK_K;

    for (size_t m = 0; m < CountM; ++m) {
        for (size_t n = 0; n < CountN; ++n) {

            float sum = 0.0f;

            const block_q8_K* a_blocks = reinterpret_cast<const block_q8_K*>(QuantA + m * BlkCountK * sizeof(block_q8_K));
            const block_tq1_0* b_blocks = reinterpret_cast<const block_tq1_0*>(QuantB + n * BlkCountK * sizeof(block_tq1_0));

            for (size_t blk = 0; blk < BlkCountK; ++blk) {
                const block_q8_K& a_block = a_blocks[blk];
                const block_tq1_0& b_block = b_blocks[blk];

                for (size_t k = 0; k < QK_K; ++k) {
                    int8_t a_val = a_block.qs[k];
                    uint8_t b_val = b_block.qs[k];

                    // b_val is 2-bit ternary (0/1/2), map back to [-1, 0, +1]
                    int b_ternary = 0;
                    if (b_val == 0) {
                        b_ternary = -1;
                    } else if (b_val == 1) {
                        b_ternary = 0;
                    } else if (b_val == 2) {
                        b_ternary = 1;
                    }

                    sum += static_cast<float>(a_val) * static_cast<float>(b_ternary);
                }

                sum *= a_block.d * static_cast<float>(b_block.d);
            }

            if (Bias != nullptr) {
                sum += Bias[n];
            }

            C[m * ldc + n] = sum;
        }
    }

    return CountM;
}
// Kernel dispatch table
const MLAS_QNBIT_GEMM_DISPATCH MlasSQTernaryBitGemmDispatchFallback = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q2BitGemmPackQuantBDataSize = QTernaryBitGemmPackQuantBDataSize;
    d.SQ2BitGemmPackQuantBData = nullptr;
    d.Q2BitGemmPerGemmWorkspaceSize = QTernaryBitGemmPerGemmWorkspaceSize;
    d.SQ2BitGemmKernel_CompInt8 = SQTernaryBitGemmKernel_TQ1_0_Q8_K; // Not implemented in fallback
    d.QuantizeARow_CompInt8 = QuantizeARow_Q8_K;

    return d;
}();
