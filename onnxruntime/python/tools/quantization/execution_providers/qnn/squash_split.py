# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define Split squash."""

import onnx

from ... import fusions, onnx_model


class SquashSplit(fusions.Fusion):
    """Squash for Split."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "", "Split")

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Squash redundant Split.

        Split can be squashed if and only if it has only one output.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Split).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """
        if len(node.output) > 1:
            return

        for child in self.model.get_children(node, input_name_to_nodes=input_name_to_nodes):
            for idx in range(len(child.input)):
                if child.input[idx] == node.output[0]:
                    child.input[idx] = node.input[0]

        self.nodes_to_remove.append(node)
