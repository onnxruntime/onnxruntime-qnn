# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define Mul+Add fusion to BatchNorm."""

import onnx
from onnx import helper, TensorProto

from ... import fusions, onnx_model


class FusionMulAdd(fusions.Fusion):
    """Fusion for Mul + Add."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        # Search for Add nodes, as Add is the last operation in the sequence.
        # The Mul node is expected to precede the Add node.
        super().__init__(model, "BatchNormalization", "Add")

    def _fuse_mul_add_to_batchnorm(
        self,
        add_node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Fuse a Mul + Add sequence into a BatchNorm node.

        Pattern:

            Input (N, C, H, W)
                |
               Mul (Input, Scale_Tensor)
                |
               Add (Mul_Output, Bias_Tensor)
                |
            Output (N, C, H, W)

        This sequence can be fused into a single BatchNormalization with:
        - scale = Scale_Tensor (second input of Mul)
        - B = Bias_Tensor (second input of Add)
        - mean = 0
        - var = 1
        - epsilon = 0
        - momentum = 0
        """
        # Ensure the Add node has two inputs
        if len(add_node.input) != 2:
            return False

        # Check if the first input of Add comes from a Mul node
        if add_node.input[0] not in output_name_to_node:
            return False

        mul_node = output_name_to_node[add_node.input[0]]
        if mul_node.op_type != "Mul":
            return False

        # Ensure the Mul node has two inputs
        if len(mul_node.input) != 2:
            return False

        # Get the second input of Mul (scale) and Add (bias)
        # These must be initializer tensors (constants)
        scale_tensor_name = mul_node.input[1]
        bias_tensor_name = add_node.input[1]

        scale_initializer = self.model.get_initializer(scale_tensor_name)
        bias_initializer = self.model.get_initializer(bias_tensor_name)

        if scale_initializer is None or bias_initializer is None:
            # One or both are not initializers, cannot fuse
            return False

        # Get input tensor of Mul (which is the effective input for BatchNorm)
        bn_input_name = mul_node.input[0]
        bn_output_name = add_node.output[0]

        # Get shapes for broadcast validation
        input_shape = self.model.get_tensor_shape(bn_input_name)
        scale_shape = self.model.get_tensor_shape(scale_tensor_name)
        bias_shape = self.model.get_tensor_shape(bias_tensor_name)

        if input_shape is None or scale_shape is None or bias_shape is None:
            return False

        # For BatchNorm, scale/bias need to match the channel dimension (C) of the input
        # (N, C), (N, C, D1), (N, C, D1, D2), ...
        num_channels = input_shape[1]
        rank = len(input_shape)
        expect_shape = [num_channels] + [1]* (rank - 2)
        if scale_shape != expect_shape:
            return False
        if bias_shape != expect_shape:
            return False

        # Create constant initializers for mean and var
        mean_data = [0.0] * num_channels
        var_data = [1.0] * num_channels

        mean_initializer = helper.make_tensor(
            name=mul_node.name+"_qnn_preproc_fusion_mul_add_mean",
            data_type=TensorProto.FLOAT,
            dims=[num_channels],
            vals=mean_data,
        )
        var_initializer = helper.make_tensor(
            name=mul_node.name+"_qnn_preproc_fusion_mul_add_var",
            data_type=TensorProto.FLOAT,
            dims=[num_channels],
            vals=var_data,
        )

        self.model.add_initializer(mean_initializer)
        self.model.add_initializer(var_initializer)

        # Create the BatchNormalization node
        bn_node = onnx.helper.make_node(
            self.fused_op_type,
            name=mul_node.name+"_qnn_preproc_fusion_mul_add_bn",
            inputs=[
                bn_input_name,            # X (input)
                scale_tensor_name,        # Scale (from Mul's second input)
                bias_tensor_name,         # B (from Add's second input)
                mean_initializer.name,    # Mean (constant 0)
                var_initializer.name      # Variance (constant 1)
            ],
            outputs=[bn_output_name],
            epsilon=0.0,  # As per request
            momentum=0.0, # As per request
        )

        # Mark nodes for removal
        self.nodes_to_remove.extend([mul_node, add_node])

        # Add the new BatchNorm node
        self.nodes_to_add.append(bn_node)

        # Remove the constant inputs if they are not used by other nodes
        # This is important to clean up the graph.
        if scale_tensor_name not in input_name_to_nodes or len(input_name_to_nodes[scale_tensor_name]) == 1:
            self.model.remove_initializer(scale_tensor_name)
        if bias_tensor_name not in input_name_to_nodes or len(input_name_to_nodes[bias_tensor_name]) == 1:
            self.model.remove_initializer(bias_tensor_name)

        return True

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Fuse a sequence of Mul and Add nodes into a single BatchNorm node.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Add).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """
        # The base class search finds the 'Add' node.
        self._fuse_mul_add_to_batchnorm(node, input_name_to_nodes, output_name_to_node)