# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import filecmp
import json
import math
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import onnx
from onnx import TensorProto, helper


class QnnTensorStruct:
    def __init__(
        self, name="", onnx_data_type=TensorProto.FLOAT, is_quantized=False, scale=0.0, offset=0, dim=None, id=None
    ):
        self.name = name
        self.onnx_data_type = onnx_data_type
        self.is_quantized = is_quantized
        self.scale = scale
        self.offset = offset
        self.dim = [] if dim is None else dim
        self.id = id


def is_quantized_data_type(qnn_data_type, is_converter_json):
    if is_converter_json:
        return qnn_data_type in (0x0408, 0x0416, 0x0432, 0x0308, 0x0316, 0x0332)
    else:
        return (
            qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_8"
            or qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_16"
            or qnn_data_type == "QNN_DATATYPE_FIXED_POINT_8"
            or qnn_data_type == "QNN_DATATYPE_FIXED_POINT_16"
            or qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_32"
            or qnn_data_type == "QNN_DATATYPE_FIXED_POINT_32"
        )


def qnn_data_type_to_onnx_data_type(qnn_data_type, is_converter_json):
    if is_converter_json:
        # QNN_DATATYPE_UFIXED_POINT_8 QNN_DATATYPE_UINT_8
        if qnn_data_type == 0x0408 or qnn_data_type == 0x0108:
            return TensorProto.UINT8
        # QNN_DATATYPE_UFIXED_POINT_16 QNN_DATATYPE_UINT_16
        elif qnn_data_type == 0x0416 or qnn_data_type == 0x0116:
            return TensorProto.UINT16
        # QNN_DATATYPE_UFIXED_POINT_32 QNN_DATATYPE_UINT_32
        elif qnn_data_type == 0x0432 or qnn_data_type == 0x0132:
            return TensorProto.UINT32
        # QNN_DATATYPE_UINT_64
        elif qnn_data_type == 0x0164:
            return TensorProto.UINT64
        # QNN_DATATYPE_FIXED_POINT_8 QNN_DATATYPE_INT_8
        elif qnn_data_type == 0x0308 or qnn_data_type == 0x0008:
            return TensorProto.INT8
        # QNN_DATATYPE_FIXED_POINT_16 QNN_DATATYPE_INT_16
        elif qnn_data_type == 0x0316 or qnn_data_type == 0x0016:
            return TensorProto.INT16
        # QNN_DATATYPE_FIXED_POINT_32 QNN_DATATYPE_INT_32
        elif qnn_data_type == 0x0332 or qnn_data_type == 0x0032:
            return TensorProto.INT32
        # QNN_DATATYPE_INT_64
        elif qnn_data_type == 0x0064:
            return TensorProto.INT64
        # QNN_DATATYPE_FLOAT_16
        elif qnn_data_type == 0x0216:
            return TensorProto.FLOAT16
        # QNN_DATATYPE_FLOAT_32
        elif qnn_data_type == 0x0232:
            return TensorProto.FLOAT
        # QNN_DATATYPE_BOOL_8
        elif qnn_data_type == 0x0508:
            return TensorProto.BOOL
        else:
            raise ValueError(f"Unsupported QNN data type: {qnn_data_type!r}")
    else:
        # QNN_DATATYPE_UFIXED_POINT_8 QNN_DATATYPE_UINT_8
        if qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_8" or qnn_data_type == "QNN_DATATYPE_UINT_8":
            return TensorProto.UINT8
        # QNN_DATATYPE_UFIXED_POINT_16 QNN_DATATYPE_UINT_16
        elif qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_16" or qnn_data_type == "QNN_DATATYPE_UINT_16":
            return TensorProto.UINT16
        # QNN_DATATYPE_UFIXED_POINT_32 QNN_DATATYPE_UINT_32
        elif qnn_data_type == "QNN_DATATYPE_UFIXED_POINT_32" or qnn_data_type == "QNN_DATATYPE_UINT_32":
            return TensorProto.UINT32
        # QNN_DATATYPE_UINT_64
        elif qnn_data_type == "QNN_DATATYPE_UINT_64":
            return TensorProto.UINT64
        # QNN_DATATYPE_FIXED_POINT_8 QNN_DATATYPE_INT_8
        elif qnn_data_type == "QNN_DATATYPE_FIXED_POINT_8" or qnn_data_type == "QNN_DATATYPE_INT_8":
            return TensorProto.INT8
        # QNN_DATATYPE_FIXED_POINT_16 QNN_DATATYPE_INT_16
        elif qnn_data_type == "QNN_DATATYPE_FIXED_POINT_16" or qnn_data_type == "QNN_DATATYPE_INT_16":
            return TensorProto.INT16
        # QNN_DATATYPE_FIXED_POINT_32 QNN_DATATYPE_INT_32
        elif qnn_data_type == "QNN_DATATYPE_FIXED_POINT_32" or qnn_data_type == "QNN_DATATYPE_INT_32":
            return TensorProto.INT32
        # QNN_DATATYPE_INT_64
        elif qnn_data_type == "QNN_DATATYPE_INT_64":
            return TensorProto.INT64
        # QNN_DATATYPE_FLOAT_16
        elif qnn_data_type == "QNN_DATATYPE_FLOAT_16":
            return TensorProto.FLOAT16
        # QNN_DATATYPE_FLOAT_32
        elif qnn_data_type == "QNN_DATATYPE_FLOAT_32":
            return TensorProto.FLOAT
        # QNN_DATATYPE_BOOL_8
        elif qnn_data_type == "QNN_DATATYPE_BOOL_8":
            return TensorProto.BOOL
        else:
            raise ValueError(f"Unsupported QNN data type: {qnn_data_type!r}")


def parse_qnn_converter_json_file(qnn_convert_json, qnn_input_tensor_dic, qnn_output_tensor_dic):
    is_qnn_converter_json = True
    for qnn_tensor_name, qnn_tensor_attribute in qnn_convert_json["graph"]["tensors"].items():
        # type:0 - QNN input tensor, type:1 - QNN output tensor
        # If tensor is not IO, ignore it
        if qnn_tensor_attribute["type"] not in (0, 1):
            continue

        # Get all graph inputs & output
        qnn_tensor = QnnTensorStruct(
            name=qnn_tensor_name,
            onnx_data_type=qnn_data_type_to_onnx_data_type(qnn_tensor_attribute["data_type"], is_qnn_converter_json),
            is_quantized=is_quantized_data_type(qnn_tensor_attribute["data_type"], is_qnn_converter_json),
            dim=qnn_tensor_attribute["dims"],
            id=qnn_tensor_attribute["id"],
        )

        if (
            qnn_tensor_attribute["quant_params"]["definition"] == 1
            and qnn_tensor_attribute["quant_params"]["encoding"] == 0
        ):
            qnn_tensor.scale = qnn_tensor_attribute["quant_params"]["scale_offset"]["scale"]
            qnn_tensor.offset = -qnn_tensor_attribute["quant_params"]["scale_offset"]["offset"]

        if qnn_tensor_attribute["type"] == 0:
            qnn_input_tensor_dic[qnn_tensor_name] = qnn_tensor
        else:
            qnn_output_tensor_dic[qnn_tensor_name] = qnn_tensor

    if not qnn_input_tensor_dic or not qnn_output_tensor_dic:
        raise ValueError("Converted QNN model should have at least 1 input and 1 output.")


def external_context_reference(qnn_ctx_file, model_file_name):
    binary_path = Path(qnn_ctx_file).resolve(strict=True)
    model_directory = Path(model_file_name).resolve().parent
    try:
        relative_path = binary_path.relative_to(model_directory)
    except ValueError:
        # The EPContext loader rejects parent/absolute paths. Package one shared
        # binary beside the wrappers when the source lives outside their directory.
        relative_path = Path(binary_path.name)
        destination = model_directory / relative_path
        if ".." in relative_path.name:
            raise ValueError("External QNN context filenames must not contain '..'.") from None
        if destination.exists():
            if not filecmp.cmp(binary_path, destination, shallow=False):
                raise ValueError(f"Refusing to overwrite a different context binary: '{destination}'.") from None
        else:
            shutil.copyfile(binary_path, destination)
    if ".." in relative_path.as_posix():
        raise ValueError("External QNN context paths must not contain '..'.")
    return relative_path.as_posix()


def validate_float_io_quantization(qnn_tensor):
    bounds = {
        TensorProto.UINT8: (0, 255),
        TensorProto.INT8: (-128, 127),
        TensorProto.UINT16: (0, 65535),
        TensorProto.INT16: (-32768, 32767),
    }
    if qnn_tensor.onnx_data_type not in bounds or not math.isfinite(qnn_tensor.scale) or qnn_tensor.scale <= 0:
        raise ValueError(
            f"Tensor '{qnn_tensor.name}' requires supported 8/16-bit scale-offset quantization for float I/O. "
            "Use --quantized_IO to retain native QNN I/O."
        )
    minimum, maximum = bounds[qnn_tensor.onnx_data_type]
    if not minimum <= qnn_tensor.offset <= maximum:
        raise ValueError(f"Tensor '{qnn_tensor.name}' has a zero point outside its data type range.")


def generate_wrapper_onnx_file(
    grap_name,
    model_file_name,
    qnn_input_tensor_dic,
    qnn_output_tensor_dic,
    disable_embed_mode,
    qnn_ctx_file,
    quantized_IO,
    qnn_sdk_version="unknown",
):
    graph_nodes = []
    ini_list = []
    value_infos = []
    qnn_inputs = sorted(qnn_input_tensor_dic.values(), key=lambda tensor: tensor.id if tensor.id is not None else 0)
    qnn_outputs = sorted(qnn_output_tensor_dic.values(), key=lambda tensor: tensor.id if tensor.id is not None else 0)

    if not quantized_IO:
        for tensor in qnn_inputs + qnn_outputs:
            if tensor.is_quantized:
                validate_float_io_quantization(tensor)

    model_inputs = []
    for qnn_input in qnn_inputs:
        if qnn_input.is_quantized and not quantized_IO:
            q_scale_input_name = qnn_input.name + "_scale"
            q_offset_input_name = qnn_input.name + "_zp"
            q_scale = helper.make_tensor(q_scale_input_name, TensorProto.FLOAT, [], [qnn_input.scale])
            ini_list.append(q_scale)
            q_offset = helper.make_tensor(q_offset_input_name, qnn_input.onnx_data_type, [], [qnn_input.offset])
            ini_list.append(q_offset)
            input_name = qnn_input.name + "_dq"

            q_node = helper.make_node(
                "QuantizeLinear",
                name=qnn_input.name,
                inputs=[input_name, q_scale_input_name, q_offset_input_name],
                outputs=[qnn_input.name],
            )

            graph_nodes.append(q_node)
            model_inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, qnn_input.dim))
            value_infos.append(helper.make_tensor_value_info(qnn_input.name, qnn_input.onnx_data_type, qnn_input.dim))
        else:
            model_inputs.append(helper.make_tensor_value_info(qnn_input.name, qnn_input.onnx_data_type, qnn_input.dim))

    if disable_embed_mode:
        ep_cache_context_content = external_context_reference(qnn_ctx_file, model_file_name)
        ctx_embed_mode = 0
    else:
        with open(qnn_ctx_file, "rb") as file:
            ep_cache_context_content = file.read()
        ctx_embed_mode = 1

    qnn_ep_context_node = helper.make_node(
        "EPContext",
        name=grap_name,
        inputs=[tensor.name for tensor in qnn_inputs],
        outputs=[tensor.name for tensor in qnn_outputs],
        ep_cache_context=ep_cache_context_content,
        embed_mode=ctx_embed_mode,
        ep_sdk_version=qnn_sdk_version,
        source="Qnn",
        domain="com.microsoft",
    )
    graph_nodes.append(qnn_ep_context_node)

    model_outputs = []
    for qnn_output in qnn_outputs:
        if qnn_output.is_quantized and not quantized_IO:
            dq_scale_input_name = qnn_output.name + "_scale"
            dq_offset_input_name = qnn_output.name + "_zp"
            dq_scale = helper.make_tensor(dq_scale_input_name, TensorProto.FLOAT, [], [qnn_output.scale])
            ini_list.append(dq_scale)
            dq_offset = helper.make_tensor(dq_offset_input_name, qnn_output.onnx_data_type, [], [qnn_output.offset])
            ini_list.append(dq_offset)
            output_name = qnn_output.name + "_dq"

            dq_node = helper.make_node(
                "DequantizeLinear",
                name=output_name,
                inputs=[qnn_output.name, dq_scale_input_name, dq_offset_input_name],
                outputs=[output_name],
            )

            graph_nodes.append(dq_node)
            model_outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, qnn_output.dim))
            value_infos.append(
                helper.make_tensor_value_info(qnn_output.name, qnn_output.onnx_data_type, qnn_output.dim)
            )
        else:
            model_outputs.append(
                helper.make_tensor_value_info(qnn_output.name, qnn_output.onnx_data_type, qnn_output.dim)
            )

    graph_def = helper.make_graph(graph_nodes, "qnn-onnx-model", model_inputs, model_outputs, ini_list, "", value_infos)

    model_def = helper.make_model(
        graph_def,
        producer_name="MS",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model_def)

    onnx.save(model_def, model_file_name)


# parse Qnn graph from the json file that extracted from context binary file
def parse_qnn_graph(qnn_graph, qnn_input_tensor_dic, qnn_output_tensor_dic):
    is_qnn_converter_json = False
    graph_name = qnn_graph["info"]["graphName"]
    raw_inputs = qnn_graph["info"]["graphInputs"]
    raw_outputs = qnn_graph["info"]["graphOutputs"]

    for tensor_id, raw_input in enumerate(raw_inputs):
        tensor_info = raw_input["info"]
        qnn_tensor = QnnTensorStruct()
        qnn_tensor.id = tensor_id
        qnn_tensor.name = tensor_info["name"]
        qnn_tensor.onnx_data_type = qnn_data_type_to_onnx_data_type(tensor_info["dataType"], is_qnn_converter_json)
        qnn_tensor.is_quantized = is_quantized_data_type(tensor_info["dataType"], is_qnn_converter_json)
        qnn_tensor.dim = tensor_info["dimensions"]
        if (
            tensor_info["quantizeParams"]["definition"] == "QNN_DEFINITION_DEFINED"
            and tensor_info["quantizeParams"]["quantizationEncoding"] == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET"
        ):
            qnn_tensor.scale = tensor_info["quantizeParams"]["scaleOffset"]["scale"]
            qnn_tensor.offset = 0 - tensor_info["quantizeParams"]["scaleOffset"]["offset"]
        qnn_input_tensor_dic[qnn_tensor.name] = qnn_tensor

    for tensor_id, raw_output in enumerate(raw_outputs):
        tensor_info = raw_output["info"]
        qnn_tensor = QnnTensorStruct()
        qnn_tensor.id = tensor_id
        qnn_tensor.name = tensor_info["name"]
        qnn_tensor.onnx_data_type = qnn_data_type_to_onnx_data_type(tensor_info["dataType"], is_qnn_converter_json)
        qnn_tensor.is_quantized = is_quantized_data_type(tensor_info["dataType"], is_qnn_converter_json)
        qnn_tensor.dim = tensor_info["dimensions"]
        if (
            tensor_info["quantizeParams"]["definition"] == "QNN_DEFINITION_DEFINED"
            and tensor_info["quantizeParams"]["quantizationEncoding"] == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET"
        ):
            qnn_tensor.scale = tensor_info["quantizeParams"]["scaleOffset"]["scale"]
            qnn_tensor.offset = 0 - tensor_info["quantizeParams"]["scaleOffset"]["offset"]
        qnn_output_tensor_dic[qnn_tensor.name] = qnn_tensor

    if not qnn_input_tensor_dic or not qnn_output_tensor_dic:
        raise ValueError("Converted QNN model should have at least 1 input and 1 output.")

    return graph_name


def extract_qnn_context_json(qnn_bin, qnn_json, qnn_context_binary_utility):
    """Extract graph metadata from a QNN context binary."""
    utility_path = Path(qnn_context_binary_utility)
    if utility_path.parent != Path(".") or "/" in qnn_context_binary_utility or "\\" in qnn_context_binary_utility:
        qnn_context_binary_utility = str(utility_path.resolve())
    qnn_bin_path = Path(qnn_bin).resolve()
    qnn_json_path = Path(qnn_json).resolve()
    try:
        subprocess.run(
            [
                qnn_context_binary_utility,
                f"--context_binary={qnn_bin_path}",
                f"--json_file={qnn_json_path.name}",
            ],
            cwd=qnn_json_path.parent,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not find '{qnn_context_binary_utility}'. Add the QAIRT host tools directory to PATH "
            "or pass --qnn_context_binary_utility with its full path."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to extract metadata from QNN context binary '{qnn_bin}'.") from error

    if not qnn_json_path.is_file():
        raise RuntimeError(f"QNN context binary utility did not create the expected JSON file '{qnn_json}'.")


def load_qnn_json(qnn_bin, qnn_json, qnn_context_binary_utility):
    if qnn_json:
        with open(qnn_json, encoding="utf-8") as qnn_json_file:
            return json.load(qnn_json_file), qnn_json

    with tempfile.TemporaryDirectory(prefix="qnn_context_binary_") as temporary_directory:
        extracted_json = str(Path(temporary_directory) / "context.json")
        extract_qnn_context_json(qnn_bin, extracted_json, qnn_context_binary_utility)
        with open(extracted_json, encoding="utf-8") as qnn_json_file:
            return json.load(qnn_json_file), extracted_json


# Preserve the QNN context's tensor layout; optionally expose fixed-point I/O as
# float32 through QuantizeLinear/DequantizeLinear nodes.
def main():
    parser = ArgumentParser("Generate Onnx model which includes the QNN context binary.")
    parser.add_argument("-b", "--qnn_bin", help="Required. Path to Qnn context binary file.", required=True, type=str)
    parser.add_argument(
        "-q",
        "--qnn_json",
        help=(
            "Optional. Path to QNN graph metadata JSON generated by the converter or "
            "qnn-context-binary-utility. If omitted, metadata is extracted from --qnn_bin."
        ),
        type=str,
    )
    parser.add_argument(
        "--qnn_context_binary_utility",
        default="qnn-context-binary-utility",
        help=(
            "Path to qnn-context-binary-utility. Used only when --qnn_json is omitted. "
            "Defaults to resolving qnn-context-binary-utility from PATH."
        ),
        type=str,
    )
    parser.add_argument(
        "--disable_embed_mode",
        action="store_true",
        default=False,
        help="Set embed_mode=0 and reference a relative context binary path, copying the binary beside the model if needed.",
    )
    parser.add_argument(
        "--quantized_IO",
        action="store_true",
        default=False,
        help="QNN converted context binary use quantized data as graph inputs and outputs. Will keep it if quantized_IO=True, otherwise, will insert Q and DQ nodes accordingly to make the graph inputs & outputs as float32 data type.",
    )
    args = parser.parse_args()

    try:
        # Parse QNN graph metadata to get the graph input and output information.
        qnn_json_obj, qnn_json_path = load_qnn_json(args.qnn_bin, args.qnn_json, args.qnn_context_binary_utility)
        if "graph" in qnn_json_obj and "tensors" in qnn_json_obj["graph"]:
            print("This json file is from Qnn converter")
            qnn_input_tensor_dic = {}
            qnn_output_tensor_dic = {}
            parse_qnn_converter_json_file(qnn_json_obj, qnn_input_tensor_dic, qnn_output_tensor_dic)
            json_path = Path(qnn_json_path)
            generate_wrapper_onnx_file(
                "QnnContext",
                str(json_path.with_name(json_path.stem + "_qnn_ctx.onnx")),
                qnn_input_tensor_dic,
                qnn_output_tensor_dic,
                args.disable_embed_mode,
                args.qnn_bin,
                args.quantized_IO,
            )
        elif "info" in qnn_json_obj and "graphs" in qnn_json_obj["info"]:
            print("This json file is extracted from QNN context binary file")
            qnn_version = qnn_json_obj["info"]["buildId"]
            if not qnn_json_obj["info"]["graphs"]:
                raise ValueError("QNN metadata contains no graphs.")
            for qnn_graph in qnn_json_obj["info"]["graphs"]:
                qnn_input_tensor_dic = {}
                qnn_output_tensor_dic = {}
                graph_name = parse_qnn_graph(qnn_graph, qnn_input_tensor_dic, qnn_output_tensor_dic)
                if not graph_name or graph_name in (".", "..") or "/" in graph_name or "\\" in graph_name:
                    raise ValueError(f"QNN graph name is not a valid output filename: {graph_name!r}")
                ctx_file_name = graph_name + "_qnn_ctx.onnx"
                if not args.quantized_IO:
                    ctx_file_name = ctx_file_name.replace(".onnx", "_fp32_io.onnx")
                generate_wrapper_onnx_file(
                    graph_name,
                    ctx_file_name,
                    qnn_input_tensor_dic,
                    qnn_output_tensor_dic,
                    args.disable_embed_mode,
                    args.qnn_bin,
                    args.quantized_IO,
                    qnn_version,
                )
        else:
            raise ValueError("Unrecognized QNN metadata JSON format.")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
