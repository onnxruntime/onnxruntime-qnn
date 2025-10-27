#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from onnxruntime.quantization.shape_inference import quant_pre_process


class TestQuantPreprocess(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="ort.quant_preprocess_")
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_upsample_model(self, input_shape=(1, 3, 32, 32)):
        """
        Build a model with deprecated Upsample op (opset < 10) for testing version conversion.
        """
        input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
        output_shape = (input_shape[0], input_shape[1], input_shape[2] * 2, input_shape[3] * 2)
        output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)

        # Create scales for upsample
        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        scales_initializer = onnx.numpy_helper.from_array(scales, "scales")

        upsample_node = onnx.helper.make_node(
            "Upsample",
            ["input", "scales"],
            ["output"],
            name="upsample_node",
            mode="nearest",
        )

        graph = onnx.helper.make_graph(
            [upsample_node],
            "upsample_graph",
            [input_tensor],
            [output_tensor],
            initializer=[scales_initializer],
        )
        # Use opset 9 to trigger Upsample -> Resize conversion
        opset_imports = [onnx.helper.make_opsetid("", 9)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return model

    def test_upsample_to_resize_conversion(self):
        """
        Test that deprecated Upsample ops are converted to Resize ops.
        """
        model = self.build_upsample_model()
        input_path = self.temp_path / "input_model.onnx"
        output_path = self.temp_path / "preprocessed_model.onnx"

        onnx.save_model(model, input_path)

        # Verify original model has Upsample op
        self.assertEqual(model.graph.node[0].op_type, "Upsample")
        self.assertEqual(model.opset_import[0].version, 9)

        quant_pre_process(
            input_model=str(input_path),
            output_model_path=str(output_path),
            skip_optimization=True,
            skip_onnx_shape=True,
            skip_symbolic_shape=True,
        )

        self.assertTrue(output_path.exists())
        preprocessed_model = onnx.load(str(output_path))

        # Verify Upsample was converted to Resize and opset was upgraded
        node_types = [node.op_type for node in preprocessed_model.graph.node]
        assert "Resize" in node_types
        assert "Upsample" not in node_types
        assert preprocessed_model.opset_import[0].version >= 11


if __name__ == "__main__":
    unittest.main()
