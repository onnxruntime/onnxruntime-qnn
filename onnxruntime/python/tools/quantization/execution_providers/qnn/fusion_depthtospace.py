# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Define DepthToSpace fusion."""

import onnx

from ... import fusions, onnx_model


class FusionDepthToSpace(fusions.Fusion):
    """Fusion for DepthToSpace."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize.

        Args:
            model: An onnx_model.ONNXModel instance.
        """
        super().__init__(model, "DepthToSpace", "Reshape")

    def _get_target_child(self, parent_node, target_op_type, input_name_to_nodes):
        """Get target child of given node."""
        if parent_node.output[0] not in input_name_to_nodes:
            return None

        children = input_name_to_nodes[parent_node.output[0]]
        if len(children) > 1 or children[0].op_type != target_op_type:
            return None

        return children[0]

    def _get_tensor_shape(self, tensor_name):
        """Get shape for given tensor name."""
        tensor_type = self.model.get_tensor_type(tensor_name)
        if not tensor_type:
            return None

        tensor_shape = self.tensor_shape_to_list(tensor_type)
        if not tensor_shape:
            return None

        return tensor_shape

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Fuse a sequence of Reshape and Transpose nodes into a single DepthToSpace node.

        Spatial-Last Pattern:

                |     [N, C, H, W]
             Reshape
                |     [N, blk, blk, C / blk**2, H, W] or [N, C / blk**2, blk, blk, H, W]
            Transpose
                |     [N, C / blk**2, H, blk, W, blk]
             Reshape
                |     [N, C / blk**2, H * blk, W * blk]

        Spatial-First Pattern:

                |     [N, H, W, C]
             Reshape
                |     [N, H, W, blk, blk, C / blk**2] or [N, H, W, C / blk**2, blk, blk]
            Transpose
                |     [N, H, blk, W, blk, C / blk**2]
             Reshape
                |     [N, H * blk, W * blk, C / blk**2]

        Since ONNX DepthToSpace expects spatial-last, additionaly Transpose will be inserted around the fused node.

        Args:
            node: An onnx.NodeProto matching the specified search type (i.e., Reshape).
            input_name_to_nodes: A dict mapping tensor name to consumed nodes.
            output_name_to_node: A dict mapping tensor name to produced node.
        """
        reshape_node1 = node

        if (
            (transpose_node := self._get_target_child(reshape_node1, "Transpose", input_name_to_nodes)) is None
            or (reshape_node2 := self._get_target_child(transpose_node, "Reshape", input_name_to_nodes)) is None
        ):
            return False

        if (
            (input_shape := self._get_tensor_shape(reshape_node1.input[0])) is None
            or (reshape_shape1 := self._get_tensor_shape(reshape_node1.output[0])) is None
            or (reshape_shape2 := self._get_tensor_shape(reshape_node2.output[0])) is None
        ):
            return False

        # Check rank.
        if len(input_shape) != 4 or len(reshape_shape1) != 6 or len(reshape_shape2) != 4:
            return False

        transpose_perm = self.get_node_attribute(transpose_node, "perm")
        mode = None

        # Check for spatial-last pattern.
        batch, channel, height, width = input_shape
        blocksize = reshape_shape1[2]
        if (
            reshape_shape1 == [batch, blocksize, blocksize, channel // blocksize**2, height, width]
            and transpose_perm == [0, 3, 4, 1, 5, 2]
            and reshape_shape2 == [batch, channel // blocksize**2, height * blocksize, width * blocksize]
        ):
            mode = "DCR"
        elif (
            reshape_shape1 == [batch, channel // blocksize**2, blocksize, blocksize, height, width]
            and transpose_perm == [0, 1, 4, 2, 5, 3]
            and reshape_shape2 == [batch, channel // blocksize**2, height * blocksize, width * blocksize]
        ):
            mode = "CRD"

        if mode is not None:
            d2s_node = onnx.helper.make_node(
                self.fused_op_type,
                [reshape_node1.input[0]],
                [reshape_node2.output[0]],
                name=self.create_unique_node_name(),
                blocksize=blocksize,
                mode=mode,
            )
            self.nodes_to_add.append(d2s_node)
            self.nodes_to_remove.extend([reshape_node1, transpose_node, reshape_node2])
            return

        # Check for spatial-first pattern.
        batch, height, width, channel = input_shape
        blocksize = reshape_shape1[4]
        if (
            reshape_shape1 == [batch, height, width, blocksize, blocksize, channel // blocksize**2]
            and transpose_perm == [0, 1, 3, 2, 4, 5]
            and reshape_shape2 == [batch, height * blocksize, width * blocksize, channel // blocksize**2]
        ):
            mode = "DCR"
        elif (
            reshape_shape1 == [batch, height, width, channel // blocksize**2, blocksize, blocksize]
            and transpose_perm == [0, 1, 4, 2, 5, 3]
            and reshape_shape2 == [batch, height * blocksize, width * blocksize, channel // blocksize**2]
        ):
            mode = "CRD"
        else:
            return

        d2s_name = self.create_unique_node_name()
        pre_transpose_name = f"{d2s_name}_pre_transpose"
        post_transpose_name = f"{d2s_name}_post_transpose"

        pre_transpose_node = onnx.helper.make_node(
            "Transpose",
            [reshape_node1.input[0]],
            [pre_transpose_name],
            name=pre_transpose_name,
            perm=[0, 3, 1, 2],
        )
        d2s_node = onnx.helper.make_node(
            self.fused_op_type,
            [pre_transpose_name],
            [post_transpose_name],
            name=d2s_name,
            blocksize=blocksize,
            mode=mode,
        )
        post_transpose_node = onnx.helper.make_node(
            "Transpose",
            [post_transpose_name],
            [reshape_node2.output[0]],
            name=post_transpose_name,
            perm=[0, 2, 3, 1],
        )

        self.nodes_to_add.extend([pre_transpose_node, d2s_node, post_transpose_node])
        self.nodes_to_remove.extend([reshape_node1, transpose_node, reshape_node2])
