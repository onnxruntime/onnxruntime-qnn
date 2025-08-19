# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
"""Define ConvBN fusion."""

import onnx
import numpy as np

# Assuming these imports exist in the project structure
from ... import fusions, onnx_model


class FusionConvBN(fusions.Fusion):
    """Fusion for Convolution and BatchNormalization."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "ConvBnFusion", "Conv") # Search for 'Conv' nodes

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Fuse a Convolution node followed by a BatchNormalization node.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Conv).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """
        # Ensure the current node is a Convolution
        if node.op_type != "Conv":
            return False # This shouldn't happen if super().__init__ is set to "Conv"

        conv_node = node
        conv_output_name = conv_node.output[0]

        # Check if the output of the Conv node is consumed by a single BatchNormalization node
        if conv_output_name not in input_name_to_nodes:
            return False # Conv output not consumed by any node

        consumers = input_name_to_nodes[conv_output_name]
        if len(consumers) != 1 or consumers[0].op_type != "BatchNormalization":
            return False # Not consumed by a single BN node

        bn_node = consumers[0]

        # Get the input tensors for the BatchNormalization node
        # BN inputs: [input, scale, B, mean, var]
        if len(bn_node.input) != 5:
            # BatchNormalization expects 5 inputs: X, scale, B, mean, var
            # (X is the output of the Conv node here)
            return False

        bn_scale_name = bn_node.input[1]
        bn_b_name = bn_node.input[2]
        bn_mean_name = bn_node.input[3]
        bn_var_name = bn_node.input[4]

        # Retrieve the values of these tensors from the model's initializers
        bn_scale = self.model.get_initializer(bn_scale_name)
        bn_b = self.model.get_initializer(bn_b_name)
        bn_mean = self.model.get_initializer(bn_mean_name)
        bn_var = self.model.get_initializer(bn_var_name)

        if any(t is None for t in [bn_scale, bn_b, bn_mean, bn_var]):
            return False # One or more BN initializers not found

        # Extract epsilon from BatchNormalization node attributes
        epsilon = self.get_node_attribute(bn_node, "epsilon") # Default epsilon is 1e-5
        epsilon = 1e-5 if epsilon is None else epsilon

        # Get Conv node's weight and bias initializers
        # Conv inputs: [input, weight, bias (optional)]
        conv_weight_name = conv_node.input[1]
        conv_weight = self.model.get_initializer(conv_weight_name)
        if conv_weight is None:
            return False # Conv weight initializer not found

        conv_bias = None
        conv_bias_name = None
        if len(conv_node.input) > 2: # Check if Conv has a bias input
            conv_bias_name = conv_node.input[2]
            conv_bias = self.model.get_initializer(conv_bias_name)
            # It's okay if conv_bias is None if it's not present, we'll handle it.

        # Convert initializers to numpy arrays for calculation
        try:
            bn_scale_np = onnx.numpy_helper.to_array(bn_scale)
            bn_b_np = onnx.numpy_helper.to_array(bn_b)
            bn_mean_np = onnx.numpy_helper.to_array(bn_mean)
            bn_var_np = onnx.numpy_helper.to_array(bn_var)
            conv_weight_np = onnx.numpy_helper.to_array(conv_weight)
            conv_bias_np = onnx.numpy_helper.to_array(conv_bias) if conv_bias else np.zeros(bn_scale_np.shape)

        except Exception:
            return False # Failed to convert initializers to numpy arrays

        # --- Perform the fusion calculations ---
        # Fusing BN into Conv:
        # Output = scale * (Conv_out - mean) / sqrt(var + epsilon) + B
        # Conv_out = Conv_weight * Input + Conv_bias
        #
        # Output = scale * (Conv_weight * Input + Conv_bias - mean) / sqrt(var + epsilon) + B
        # Output = (scale / sqrt(var + epsilon)) * (Conv_weight * Input + Conv_bias - mean) + B
        # Output = (scale / sqrt(var + epsilon)) * Conv_weight * Input + (scale / sqrt(var + epsilon)) * (Conv_bias - mean) + B
        #
        # Let K = scale / sqrt(var + epsilon)
        # New_Conv_weight = K * Conv_weight
        # New_Conv_bias = K * (Conv_bias - mean) + B

        std_dev_inv = bn_scale_np / np.sqrt(bn_var_np + epsilon)

        # Reshape std_dev_inv to match weight dimensions for broadcasting
        # For Conv (N, C, H, W) weight is (out_channels, in_channels, kH, kW)
        # std_dev_inv and other BN params are (out_channels,)
        # Reshape std_dev_inv for element-wise multiplication with Conv_weight
        # It needs to be (out_channels, 1, 1, 1)
        # Assuming conv_weight_np.shape[0] is the number of output channels
        new_conv_weight_np = conv_weight_np * std_dev_inv.reshape(-1, 1, 1, 1)

        new_conv_bias_np = std_dev_inv * (conv_bias_np - bn_mean_np) + bn_b_np

        # --- Create new initializers for the fused Conv node ---
        new_conv_weight_initializer = onnx.numpy_helper.from_array(new_conv_weight_np, name=conv_weight_name)
        conv_bias_name = node.name + "_qnn_preproc_fusion_conv_bn_bias" if conv_bias_name is None else conv_bias_name
        new_conv_bias_initializer = onnx.numpy_helper.from_array(new_conv_bias_np, name=conv_bias_name)

        # Add the new bias initializer to the model if it's not already there
        # If the original Conv didn't have a bias, we're adding one.
        self.model.add_initializer(new_conv_bias_initializer)

        # Update the original Conv node's weight and bias
        # Remove old weight/bias initializers if they are no longer used by other nodes
        self.model.add_initializer(new_conv_weight_initializer) # This will replace the existing one with the same name

        # Update conv_node inputs: if no bias was present, add it.
        # Ensure it always has 3 inputs (X, W, B) after fusion
        if len(conv_node.input) < 3:
            conv_node.input.append(new_conv_bias_initializer.name)
        else:
            # Replace the existing bias input with the new one
            conv_node.input[2] = new_conv_bias_initializer.name

        # Update the output of the Conv node to point to the output of the BN node
        conv_node.output[0] = bn_node.output[0]

        # Add the BN node to the list of nodes to be removed
        self.nodes_to_remove.append(bn_node)

        # We need to explicitly remove initializers that are no longer needed
        # (bn_scale, bn_b, bn_mean, bn_var)
        self.model.remove_initializer(bn_scale.name)
        self.model.remove_initializer(bn_b.name)
        self.model.remove_initializer(bn_mean.name)
        self.model.remove_initializer(bn_var.name)

        # Mark the fusion as successful
        return True
