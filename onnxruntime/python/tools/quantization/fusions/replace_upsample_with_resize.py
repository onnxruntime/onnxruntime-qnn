# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from . import fusions, onnx_model


class ReplaceUpsampleWithResize(fusions.Fusion):
    """Replace Upsample with Resize."""

    def __init__(self, model: onnx_model.ONNXModel):
        """Initialize."""
        super().__init__(model, "Resize", "Upsample")

    def fuse(
        self,
        node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """Replace Upsample with Resize."""
        new_nodes = []
        for node in model.model.graph.node:
            if node.op_type == "Upsample":
                mode = None
                for attr in node.attribute:
                    if attr.name == 'mode':
                        mode = attr.s.decode('utf-8')
                        break

                scales_input = None
                if model.model.opset_import[0].version > 7:
                    scales_input = node.input[1] if len(node.input) > 1 else ''
                    resize_inputs = [node.input[0], node.name + '_roi', scales_input]
                else:
                    if model.model.opset_import[0].version == 7:
                        for attr in node.attribute:
                            if attr.name == 'scales':
                                scales_input = attr.floats
                                break

                        scales_input = np.array(list(scales_input), np.float32)
                    else:
                        h_scale = 1
                        w_scale = 1
                        for attr in node.attribute:
                            if attr.name == 'height_scale':
                                h_scale = attr.float
                            elif attr.name == 'width_scale':
                                w_scale = attr.float

                    scales_tensor = onnx.helper.make_tensor(
                        name=node.name + '_scales',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=scales_input.shape,
                        vals=scales_input.flatten().tolist()
                    )
                    model.model.graph.initializer.append(scales_tensor)

                    resize_inputs = [node.input[0], node.name + '_roi', node.name + '_scales']

                roi_tensor = onnx.helper.make_tensor(
                    name=node.name + '_roi',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=(len(scales_input)*2,),
                    vals=[0]*len(scales_input)+[1]*len(scales_input)
                )
                model.model.graph.initializer.append(roi_tensor)

                resize_node = onnx.helper.make_node(
                    op_type='Resize',
                    inputs=resize_inputs,
                    outputs = node.output,
                    mode=mode,
                    nearest_mode='floor'
                )
                # node = resize_node

            # new_nodes.append(node)

        # model.model.graph.ClearField('node')
        # model.model.graph.node.extend(new_nodes)

                self.nodes_to_remove.append(node)
                self.nodes_to_add.append(resize_node)

        return True

    def apply(self) -> bool:
        """Apply."""
        if super().apply():
            self.model.topological_sort()
            return True
        return False
