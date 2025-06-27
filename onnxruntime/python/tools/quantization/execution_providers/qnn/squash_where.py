# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define Where squash."""

import numpy as np
import onnx

from ... import fusions, onnx_model


class SquashWhere(fusions.Fusion):
    """Squash for Where."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "", "Where")

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Squash redundant Where.

        Where can be squashed if its condition is Constant and has values all True or False.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Where).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """
        condition = self.model.get_constant_value(node.input[0])
        if condition is None:
            return

        rerouted_input = None
        if np.all(condition):
            rerouted_input = node.input[1]
        elif np.all(np.logical_not(condition)):
            rerouted_input = node.input[2]

        if rerouted_input is None:
            return

        for child in self.model.get_children(node, input_name_to_nodes=input_name_to_nodes):
            for idx in range(len(child.input)):
                if child.input[idx] == node.output[0]:
                    child.input[idx] = rerouted_input

        self.nodes_to_remove.append(node)
