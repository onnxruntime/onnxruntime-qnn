# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT_PATH = Path(__file__).parents[1] / "gen_qnn_ctx_onnx_model.py"
_SPEC = importlib.util.spec_from_file_location("gen_qnn_ctx_onnx_model", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestGenQnnCtxOnnxModel(unittest.TestCase):
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
