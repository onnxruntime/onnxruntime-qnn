# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

_SCRIPT_PATH = Path(__file__).parents[1] / "gen_qnn_ctx_onnx_model.py"
_SPEC = importlib.util.spec_from_file_location("gen_qnn_ctx_onnx_model", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestGenQnnCtxOnnxModel(unittest.TestCase):
    @staticmethod
    def _context_tensor(name, data_type="QNN_DATATYPE_FLOAT_32", scale=0.25, offset=-3):
        return {
            "info": {
                "name": name,
                "dataType": data_type,
                "dimensions": [1, 2],
                "quantizeParams": {
                    "definition": "QNN_DEFINITION_DEFINED",
                    "quantizationEncoding": "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET",
                    "scaleOffset": {"scale": scale, "offset": offset},
                },
            }
        }

    @classmethod
    def _context_metadata(cls, data_type="QNN_DATATYPE_FLOAT_32"):
        return {
            "info": {
                "buildId": "test-sdk",
                "graphs": [
                    {
                        "info": {
                            "graphName": "graph",
                            "graphInputs": [cls._context_tensor("z_input", data_type), cls._context_tensor("a_input")],
                            "graphOutputs": [
                                cls._context_tensor("z_output", data_type),
                                cls._context_tensor("a_output"),
                            ],
                        }
                    }
                ],
            }
        }

    @staticmethod
    def _converter_metadata():
        tensors = {}
        for name, tensor_id, tensor_type in [
            ("z_input", 2, 0),
            ("a_input", 1, 0),
            ("z_output", 4, 1),
            ("a_output", 3, 1),
        ]:
            tensors[name] = {
                "type": tensor_type,
                "data_type": 0x0232,
                "dims": [1, 2],
                "id": tensor_id,
                "quant_params": {"definition": 0, "encoding": 0},
            }
        return {"graph": {"tensors": tensors}}

    def _run_cli(self, directory, metadata, *extra_args, nested_metadata=False):
        binary = directory / "model.bin"
        binary.write_bytes(b"test context binary")
        metadata_directory = directory / "metadata" if nested_metadata else directory
        metadata_directory.mkdir(exist_ok=True)
        json_path = metadata_directory / "model.json"
        json_path.write_text(json.dumps(metadata), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), "-b", str(binary), "-q", str(json_path), *extra_args],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _ep_context(model):
        return next(node for node in model.graph.node if node.op_type == "EPContext")

    def test_context_metadata_generates_multi_io_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            result = self._run_cli(directory, self._context_metadata())
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(directory / "graph_qnn_ctx_fp32_io.onnx")
            self.assertEqual(list(self._ep_context(model).input), ["z_input", "a_input"])
            self.assertEqual(list(self._ep_context(model).output), ["z_output", "a_output"])
            onnx.checker.check_model(model)

    def test_converter_io_order_matches_ep_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            result = self._run_cli(directory, self._converter_metadata())
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(directory / "model_qnn_ctx.onnx")
            self.assertEqual(list(self._ep_context(model).input), [value.name for value in model.graph.input])
            self.assertEqual(list(self._ep_context(model).output), [value.name for value in model.graph.output])

    def test_model_declares_supported_ir_and_domains(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            result = self._run_cli(directory, self._converter_metadata())
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(directory / "model_qnn_ctx.onnx")
            self.assertEqual(
                {opset.domain: opset.version for opset in model.opset_import}, {"": 21, "com.microsoft": 1}
            )
            self.assertEqual(model.ir_version, 10)
            onnx.checker.check_model(model)

    def test_external_context_is_portable_from_nested_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            result = self._run_cli(directory, self._converter_metadata(), "--disable_embed_mode", nested_metadata=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            model_path = directory / "metadata" / "model_qnn_ctx.onnx"
            model = onnx.load(model_path)
            attributes = {attribute.name: attribute for attribute in self._ep_context(model).attribute}
            reference = attributes["ep_cache_context"].s.decode()
            self.assertFalse(Path(reference).is_absolute())
            self.assertNotIn("..", reference)
            self.assertEqual((model_path.parent / reference).read_bytes(), (directory / "model.bin").read_bytes())
            self.assertEqual(attributes["embed_mode"].i, 0)

    def test_unrecognized_metadata_exits_with_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._run_cli(Path(temporary_directory), {"unexpected": {}})
            self.assertNotEqual(result.returncode, 0)

    def test_quantized_context_float_io_preserves_type_and_quantization(self):
        for qnn_type, onnx_type in [
            ("QNN_DATATYPE_UFIXED_POINT_8", TensorProto.UINT8),
            ("QNN_DATATYPE_FIXED_POINT_8", TensorProto.INT8),
            ("QNN_DATATYPE_UFIXED_POINT_16", TensorProto.UINT16),
            ("QNN_DATATYPE_FIXED_POINT_16", TensorProto.INT16),
        ]:
            with self.subTest(qnn_type=qnn_type), tempfile.TemporaryDirectory() as temporary_directory:
                directory = Path(temporary_directory)
                metadata = self._context_metadata(qnn_type)
                output = metadata["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]
                output["quantizeParams"]["scaleOffset"] = {"scale": 0.5, "offset": -2}
                result = self._run_cli(directory, metadata)
                self.assertEqual(result.returncode, 0, result.stderr)
                model = onnx.load(directory / "graph_qnn_ctx_fp32_io.onnx")
                onnx.checker.check_model(model)
                q_node, _, dq_node = model.graph.node
                self.assertEqual(q_node.op_type, "QuantizeLinear")
                self.assertEqual(dq_node.op_type, "DequantizeLinear")
                initializers = {value.name: value for value in model.graph.initializer}
                self.assertEqual(initializers[q_node.input[2]].data_type, onnx_type)
                self.assertEqual(initializers[dq_node.input[2]].data_type, onnx_type)
                self.assertEqual(numpy_helper.to_array(initializers[q_node.input[1]]).item(), 0.25)
                self.assertEqual(numpy_helper.to_array(initializers[q_node.input[2]]).item(), 3)
                self.assertEqual(numpy_helper.to_array(initializers[dq_node.input[1]]).item(), 0.5)
                self.assertEqual(numpy_helper.to_array(initializers[dq_node.input[2]]).item(), 2)
                # Execute the generated Q/DQ boundary independently of the opaque binary.
                q_graph = onnx.helper.make_graph(
                    [q_node],
                    "quantize-boundary",
                    [model.graph.input[0]],
                    [onnx.helper.make_tensor_value_info(q_node.output[0], onnx_type, [1, 2])],
                    [initializers[name] for name in q_node.input[1:]],
                )
                q_model = onnx.helper.make_model(q_graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
                actual = ReferenceEvaluator(q_model).run(None, {q_node.input[0]: np.array([[1.0, 2.0]], np.float32)})[0]
                np.testing.assert_array_equal(actual, [[7, 11]])
                dq_graph = onnx.helper.make_graph(
                    [dq_node],
                    "dequantize-boundary",
                    [onnx.helper.make_tensor_value_info(dq_node.input[0], onnx_type, [1, 2])],
                    [model.graph.output[0]],
                    [initializers[name] for name in dq_node.input[1:]],
                )
                dq_model = onnx.helper.make_model(dq_graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
                actual = ReferenceEvaluator(dq_model).run(None, {dq_node.input[0]: actual})[0]
                np.testing.assert_array_equal(actual, [[2.5, 4.5]])

    def test_native_quantized_io_does_not_insert_qdq(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            result = self._run_cli(directory, self._context_metadata("QNN_DATATYPE_UFIXED_POINT_16"), "--quantized_IO")
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(directory / "graph_qnn_ctx.onnx")
            self.assertEqual(len(model.graph.node), 1)
            self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, TensorProto.UINT16)
            onnx.checker.check_model(model)

    def test_converter_quantized_io_uses_tensor_ids_and_signed_zero_points(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = self._converter_metadata()
            for name, data_type, scale, offset in [
                ("z_input", 0x0416, 0.125, -4),
                ("z_output", 0x0316, 0.25, 5),
            ]:
                tensor = metadata["graph"]["tensors"][name]
                tensor["data_type"] = data_type
                tensor["quant_params"] = {
                    "definition": 1,
                    "encoding": 0,
                    "scale_offset": {"scale": scale, "offset": offset},
                }
            result = self._run_cli(directory, metadata)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = onnx.load(directory / "model_qnn_ctx.onnx")
            self.assertEqual(list(self._ep_context(model).input), ["a_input", "z_input"])
            self.assertEqual(list(self._ep_context(model).output), ["a_output", "z_output"])
            initializers = {value.name: value for value in model.graph.initializer}
            self.assertEqual(initializers["z_input_zp"].data_type, TensorProto.UINT16)
            self.assertEqual(numpy_helper.to_array(initializers["z_input_zp"]).item(), 4)
            self.assertEqual(initializers["z_output_zp"].data_type, TensorProto.INT16)
            self.assertEqual(numpy_helper.to_array(initializers["z_output_zp"]).item(), -5)
            onnx.checker.check_model(model)

    def test_unsupported_quantization_requires_native_io(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = self._context_metadata("QNN_DATATYPE_UFIXED_POINT_8")
            tensor = metadata["info"]["graphs"][0]["info"]["graphInputs"][0]["info"]
            tensor["quantizeParams"]["quantizationEncoding"] = "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET"
            result = self._run_cli(directory, metadata)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use --quantized_IO", result.stderr)
            result = self._run_cli(directory, metadata, "--quantized_IO")
            self.assertEqual(result.returncode, 0, result.stderr)
            onnx.checker.check_model(onnx.load(directory / "graph_qnn_ctx.onnx"))

    def test_invalid_scale_or_zero_point_is_rejected(self):
        for scale_offset in [{"scale": 0.0, "offset": -3}, {"scale": 0.25, "offset": -256}]:
            with self.subTest(scale_offset=scale_offset), tempfile.TemporaryDirectory() as temporary_directory:
                metadata = self._context_metadata("QNN_DATATYPE_UFIXED_POINT_8")
                metadata["info"]["graphs"][0]["info"]["graphInputs"][0]["info"]["quantizeParams"]["scaleOffset"] = (
                    scale_offset
                )
                result = self._run_cli(Path(temporary_directory), metadata)
                self.assertNotEqual(result.returncode, 0)

    def test_fixed_point_32_requires_native_io(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = self._context_metadata("QNN_DATATYPE_FIXED_POINT_32")
            result = self._run_cli(directory, metadata)
            self.assertNotEqual(result.returncode, 0)
            result = self._run_cli(directory, metadata, "--quantized_IO")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_data_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._run_cli(Path(temporary_directory), self._context_metadata("QNN_DATATYPE_UNKNOWN"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported QNN data type", result.stderr)

    def test_empty_graphs_and_missing_io_are_rejected(self):
        metadata = self._context_metadata()
        metadata["info"]["graphs"][0]["info"]["graphOutputs"] = []
        for invalid in [metadata, {"info": {"buildId": "test", "graphs": []}}, {"graph": {"tensors": {}}}]:
            with self.subTest(metadata=invalid), tempfile.TemporaryDirectory() as temporary_directory:
                result = self._run_cli(Path(temporary_directory), invalid)
                self.assertNotEqual(result.returncode, 0)

    def test_external_copy_does_not_overwrite_different_binary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata_directory = directory / "metadata"
            metadata_directory.mkdir()
            destination = metadata_directory / "model.bin"
            destination.write_bytes(b"existing unrelated binary")
            result = self._run_cli(directory, self._converter_metadata(), "--disable_embed_mode", nested_metadata=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(destination.read_bytes(), b"existing unrelated binary")

    def test_multiple_graphs_share_external_binary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata = self._context_metadata()
            second = json.loads(json.dumps(metadata["info"]["graphs"][0]))
            second["info"]["graphName"] = "second_graph"
            metadata["info"]["graphs"].append(second)
            result = self._run_cli(directory, metadata, "--disable_embed_mode")
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ["graph", "second_graph"]:
                model = onnx.load(directory / f"{name}_qnn_ctx_fp32_io.onnx")
                node = self._ep_context(model)
                self.assertEqual(node.name, name)
                self.assertEqual(
                    next(attribute.s for attribute in node.attribute if attribute.name == "ep_cache_context"),
                    b"model.bin",
                )
                onnx.checker.check_model(model)

    def test_automatic_extraction_generates_embedded_model(self):
        metadata = self._context_metadata()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            binary = directory / "model.bin"
            binary.write_bytes(b"embedded context")
            original_directory = Path.cwd()

            def extract(qnn_bin, qnn_json, utility):
                Path(qnn_json).write_text(json.dumps(metadata), encoding="utf-8")

            try:
                os.chdir(directory)
                with (
                    mock.patch.object(sys, "argv", [str(_SCRIPT_PATH), "-b", str(binary)]),
                    mock.patch.object(_MODULE, "extract_qnn_context_json", side_effect=extract),
                ):
                    _MODULE.main()
            finally:
                os.chdir(original_directory)
            model = onnx.load(directory / "graph_qnn_ctx_fp32_io.onnx")
            attributes = {attribute.name: attribute for attribute in self._ep_context(model).attribute}
            self.assertEqual(attributes["embed_mode"].i, 1)
            self.assertEqual(attributes["ep_cache_context"].s, binary.read_bytes())
            self.assertEqual(attributes["ep_sdk_version"].s, b"test-sdk")
            onnx.checker.check_model(model)

    def test_extraction_reports_missing_tool_and_missing_output(self):
        for exception, message in [(FileNotFoundError(), "Could not find"), (None, "did not create")]:
            with (
                self.subTest(exception=exception),
                tempfile.TemporaryDirectory() as temporary_directory,
                mock.patch.object(_MODULE.subprocess, "run", side_effect=exception),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                _MODULE.extract_qnn_context_json(
                    "model.bin", str(Path(temporary_directory) / "context.json"), "utility"
                )

    def test_extract_qnn_context_json_runs_utility(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "context.json"

            def create_output(*args, **kwargs):
                output_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(_MODULE.subprocess, "run", side_effect=create_output) as run_mock:
                _MODULE.extract_qnn_context_json("model.bin", str(output_path), "utility")

            run_mock.assert_called_once_with(
                ["utility", f"--context_binary={Path('model.bin').resolve()}", "--json_file=context.json"],
                cwd=output_path.parent,
                check=True,
            )

    def test_extract_qnn_context_json_resolves_relative_utility_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "context.json"
            utility_path = Path("tools") / "utility"

            def create_output(*args, **kwargs):
                output_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(_MODULE.subprocess, "run", side_effect=create_output) as run_mock:
                _MODULE.extract_qnn_context_json("model.bin", str(output_path), str(utility_path))

            self.assertEqual(run_mock.call_args.args[0][0], str(utility_path.resolve()))

    def test_extract_qnn_context_json_reports_utility_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "context.json"
            error = subprocess.CalledProcessError(1, ["utility"])
            with (
                mock.patch.object(_MODULE.subprocess, "run", side_effect=error),
                self.assertRaisesRegex(RuntimeError, "Failed to extract metadata"),
            ):
                _MODULE.extract_qnn_context_json("model.bin", str(output_path), "utility")

    def test_extract_qnn_context_json_resolves_current_directory_utility(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "context.json"

            def create_output(*args, **kwargs):
                output_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(_MODULE.subprocess, "run", side_effect=create_output) as run_mock:
                _MODULE.extract_qnn_context_json("model.bin", str(output_path), "." + os.sep + "utility")
            self.assertEqual(run_mock.call_args.args[0][0], str(Path("utility").resolve()))

    def test_load_qnn_json_preserves_utf8_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "metadata.json"
            expected = {"name": "輸入"}
            json_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
            actual, _ = _MODULE.load_qnn_json("model.bin", str(json_path), "utility")
            self.assertEqual(actual, expected)

    def test_load_qnn_json_uses_supplied_metadata(self):
        expected = {"graph": {"tensors": {}}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "model_net.json"
            json_path.write_text(json.dumps(expected), encoding="utf-8")
            with mock.patch.object(_MODULE, "extract_qnn_context_json") as extract_mock:
                actual, actual_path = _MODULE.load_qnn_json("model.bin", str(json_path), "utility")

        self.assertEqual(actual, expected)
        self.assertEqual(actual_path, str(json_path))
        extract_mock.assert_not_called()

    def test_load_qnn_json_extracts_metadata_when_json_is_omitted(self):
        expected = {"info": {"buildId": "test", "graphs": []}}

        def create_output(qnn_bin, qnn_json, utility):
            self.assertEqual(qnn_bin, "model.bin")
            self.assertEqual(utility, "utility")
            Path(qnn_json).write_text(json.dumps(expected), encoding="utf-8")

        with mock.patch.object(_MODULE, "extract_qnn_context_json", side_effect=create_output):
            actual, extracted_path = _MODULE.load_qnn_json("model.bin", None, "utility")

        self.assertEqual(actual, expected)
        self.assertFalse(Path(extracted_path).exists())

    def test_main_allows_qnn_json_to_be_omitted(self):
        metadata = {"info": {"buildId": "test", "graphs": [{}]}}
        with (
            mock.patch.object(sys, "argv", ["gen_qnn_ctx_onnx_model.py", "-b", "model.bin"]),
            mock.patch.object(_MODULE, "load_qnn_json", return_value=(metadata, "temporary.json")) as load_mock,
            mock.patch.object(_MODULE, "parse_qnn_graph", return_value="graph"),
            mock.patch.object(_MODULE, "generate_wrapper_onnx_file") as generate_mock,
        ):
            _MODULE.main()

        load_mock.assert_called_once_with("model.bin", None, "qnn-context-binary-utility")
        generate_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
