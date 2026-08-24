# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""
Generate two ONNX models containing a single MyAdd UDO node:
  - myadd_fp32.onnx  : float32 model (for QNN CPU backend)
  - myadd_qdq.onnx   : uint8 QDQ model  (DQ -> MyAdd -> Q, for QNN HTP backend)

MyAdd computes: output = input + constant
  input  : shape [1, 32], float32
  output : shape [1, 32], float32
  constant attr : float, default 2.0

Usage:
  python gen_myadd_model.py [--constant 2.0] [--outdir .]
"""

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DOMAIN = "example"
OP_TYPE = "MyAdd"
INPUT_SHAPE = [1, 32]


def make_fp32_model(constant: float) -> onnx.ModelProto:
    """Float32 model: input -> MyAdd -> output."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, INPUT_SHAPE)
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, INPUT_SHAPE)

    constant_attr = helper.make_attribute("constant", constant)
    node = helper.make_node(OP_TYPE, inputs=["input"], outputs=["output"], domain=DOMAIN)
    node.attribute.append(constant_attr)

    graph = helper.make_graph([node], "myadd_fp32", [X], [Y])
    opset = helper.make_opsetid(DOMAIN, 1)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    return model


def make_qdq_model(constant: float) -> onnx.ModelProto:
    """
    QDQ model for HTP backend:
      input (f32) -> QuantizeLinear -> input_q (u8) -> DequantizeLinear -> input_dq (f32)
      -> MyAdd -> output_dq (f32) -> QuantizeLinear -> output_q (u8) -> DequantizeLinear -> output (f32)

    Scale/zero-point are computed from [-1, 1] range matching udo_op_test.cc test input.
    """
    # Input quantization: [-1, 1] range -> uint8 (scale=2/255, zp=128 so 0.0 maps to 128)
    scale_in = np.float32(2.0 / 255.0)
    zp_in = np.uint8(128)

    # Output quantization: [0, 4] range (conservatively covers input[-1,1]+constant[2.0])
    # zp=0 (asymmetric, all values positive)
    scale_out = np.float32(4.0 / 255.0)
    zp_out = np.uint8(0)

    def quant_init(name: str, val) -> onnx.TensorProto:
        return numpy_helper.from_array(np.array(val), name=name)

    # scale/zero_point initializers
    inits = [
        quant_init("scale_in",  scale_in),
        quant_init("zp_in",     zp_in),
        quant_init("scale_out", scale_out),
        quant_init("zp_out",    zp_out),
    ]

    # Value infos
    f32 = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, INPUT_SHAPE)
    u8 = lambda name: helper.make_tensor_value_info(name, TensorProto.UINT8, INPUT_SHAPE)

    input_vi = f32("input")
    output_vi = f32("output")

    # Nodes: Q -> DQ -> MyAdd -> Q -> DQ
    q_in = helper.make_node("QuantizeLinear", ["input", "scale_in", "zp_in"], ["input_q"], axis=None)
    dq_in = helper.make_node("DequantizeLinear", ["input_q", "scale_in", "zp_in"], ["input_dq"], axis=None)

    constant_attr = helper.make_attribute("constant", constant)
    myadd = helper.make_node(OP_TYPE, ["input_dq"], ["output_dq"], domain=DOMAIN)
    myadd.attribute.append(constant_attr)

    q_out = helper.make_node("QuantizeLinear", ["output_dq", "scale_out", "zp_out"], ["output_q"], axis=None)
    dq_out = helper.make_node("DequantizeLinear", ["output_q", "scale_out", "zp_out"], ["output"], axis=None)

    graph = helper.make_graph(
        [q_in, dq_in, myadd, q_out, dq_out],
        "myadd_qdq",
        [input_vi],
        [output_vi],
        initializer=inits,
    )
    onnx_opset = helper.make_opsetid("", 21)
    custom_opset = helper.make_opsetid(DOMAIN, 1)
    model = helper.make_model(graph, opset_imports=[onnx_opset, custom_opset])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate MyAdd UDO ONNX models")
    parser.add_argument("--constant", type=float, default=2.0,
                        help="Value added to each input element (default: 2.0)")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    fp32_path = os.path.join(args.outdir, "myadd_fp32.onnx")
    qdq_path = os.path.join(args.outdir, "myadd_qdq.onnx")

    onnx.save(make_fp32_model(args.constant), fp32_path)
    print(f"Saved {fp32_path}")

    onnx.save(make_qdq_model(args.constant), qdq_path)
    print(f"Saved {qdq_path}")


if __name__ == "__main__":
    main()
