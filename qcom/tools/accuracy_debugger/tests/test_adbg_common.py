# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for ``common.py`` and ``log.py``."""

from __future__ import annotations

import logging
import logging.handlers
import pickle
import re
from datetime import datetime
from pathlib import Path

import log as adbg_log
import numpy as np
import pytest
from common import (
    CSV_TO_JSON_FIELDS_MAP,
    LAYERWISE_SKIP_OPS,
    MAX_FILE_NAME_BYTES,
    QDQ_OPS_INDEX_DEFAULT,
    ActivationStatus,
    Algorithm,
    CustomException,
    DebuggerConfig,
    StrictModel,
    SubgraphExtractionError,
    VerificationError,
    create_working_directory,
    create_working_directory_with_timestamp,
    sanitize_tensor_name,
    to_py_scalar,
)
from log import (
    LOG_FILE_SUFFIX,
    LOG_FORMAT,
    configure_worker_logging,
    get_logger,
    resolve_log_level,
)
from pydantic import ValidationError

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_VERBOSE_WARNING = "WARNING!: 'VERBOSE' is not a standard log level. Using 'DEBUG' instead.\n"


def _marked(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, "_adbg_handler", False)]


# --------------------------------------------------------------------------- sanitize_tensor_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("conv1", "conv1"),
        ("_x", "_x"),
        ("/conv1/Conv", "_conv1_Conv"),
        ("/conv1/Conv_output_0", "_conv1_Conv_output_0"),
        ("1abc", "_1abc"),
        ("a.b-c:d", "a_b_c_d"),
        ("a b", "a_b"),
        ("é", "_"),
        ("_", "_"),
        ("9", "_9"),
        (123, "_123"),
        (Path("x/y"), "x_y"),
    ],
)
def test_sanitize_tensor_name(name: object, expected: str) -> None:
    assert sanitize_tensor_name(name) == expected


def test_sanitize_tensor_name_empty_raises() -> None:
    with pytest.raises(ValueError, match="Tensor name must not be empty."):
        sanitize_tensor_name("")


# --------------------------------------------------------------------------- constants and enums


def test_algorithm_str_and_format() -> None:
    assert str(Algorithm.ONESHOT) == "oneshot"
    assert f"{Algorithm.LAYERWISE}" == "layerwise"
    assert f"{Algorithm.ONESHOT}" == "oneshot"
    assert Algorithm("layerwise") is Algorithm.LAYERWISE
    assert Algorithm.ONESHOT == "oneshot"
    assert [a.value for a in Algorithm] == ["oneshot", "layerwise"]
    with pytest.raises(ValueError):
        Algorithm("cumulative_layerwise")


def test_activation_status_constants() -> None:
    assert ActivationStatus.SKIP == "SKIP"
    assert ActivationStatus.INFERENCE_FAILURE == "INFERENCE_FAILURE"
    assert ActivationStatus.VERIFICATION_FAILURE == "VERIFICATION_FAILURE"
    assert ActivationStatus.SUCCESS == "SUCCESS"


def test_constants() -> None:
    assert CSV_TO_JSON_FIELDS_MAP == {
        "QNN_Op_Name": "qnn_op_name",
        "QNN_Op_Type": "qnn_op_type",
        "Fusion_Pattern": "fusion_pattern",
        "Original_ONNX_Nodes": "original_onnx_nodes",
        "CPU_Fallback_Reason": "cpu_fallback_reason",
        "Layer Name": "layer_name",
        "Layer Output": "layer_output",
        "Layer Status": "layer_status",
        "Layer Type": "layer_type",
        "Layer Shape": "layer_shape",
        "Source(Min, Max, Median)": "source_min_max_median",
        "Target(Min, Max, Median)": "target_min_max_median",
        "Exception": "exception",
    }
    assert MAX_FILE_NAME_BYTES == 255
    assert dict(QDQ_OPS_INDEX_DEFAULT) == {"QuantizeLinear": 1, "DequantizeLinear": 2}
    with pytest.raises(TypeError):
        QDQ_OPS_INDEX_DEFAULT["Conv"] = 3  # type: ignore[index]
    assert frozenset({"QuantizeLinear", "DequantizeLinear", "Transpose", "Identity", "Constant"}) == LAYERWISE_SKIP_OPS


# --------------------------------------------------------------------------- exceptions


def test_custom_exception_str() -> None:
    assert str(CustomException("bad")) == "bad"
    err = CustomException("bad", ValueError("inner"))
    assert str(err) == "Exception: inner Reason: bad"
    assert err.msg == "bad"
    assert isinstance(err.exception, ValueError)
    assert err.args == ("bad",)


def test_exception_hierarchy() -> None:
    assert issubclass(VerificationError, CustomException)
    assert str(VerificationError("v", KeyError("k"))) == "Exception: 'k' Reason: v"
    assert issubclass(SubgraphExtractionError, Exception)
    assert not issubclass(SubgraphExtractionError, CustomException)


# --------------------------------------------------------------------------- pydantic bases


class _Cfg(DebuggerConfig):
    count: int
    arr: np.ndarray | None = None


def test_debugger_config_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        _Cfg(count=1, extra_field=2)


def test_debugger_config_validates_assignment() -> None:
    cfg = _Cfg(count=1)
    cfg.count = "5"  # type: ignore[assignment]
    assert cfg.count == 5
    with pytest.raises(ValidationError):
        cfg.count = "not-an-int"  # type: ignore[assignment]


def test_debugger_config_allows_arbitrary_types_and_merges_config() -> None:
    cfg = _Cfg(count=1, arr=np.zeros(2))
    assert isinstance(cfg.arr, np.ndarray)
    merged = DebuggerConfig.model_config
    assert merged["extra"] == "forbid"
    assert merged["validate_assignment"] is True
    assert merged["arbitrary_types_allowed"] is True
    assert merged["protected_namespaces"] == ()
    assert merged["populate_by_name"] is True


def test_strict_model_allows_model_prefix() -> None:
    class _M(StrictModel):
        model_path: str

    assert _M(model_path="x").model_path == "x"
    with pytest.raises(ValidationError):
        _M(model_path="x", other=1)


# --------------------------------------------------------------------------- to_py_scalar


def test_to_py_scalar_float32() -> None:
    out = to_py_scalar(np.float32(0.1))
    assert type(out) is float
    assert out == 0.1
    assert to_py_scalar(np.float32(0.00474)) == 0.00474


def test_to_py_scalar_float16_and_float64() -> None:
    out = to_py_scalar(np.float16(0.1))
    assert type(out) is float
    assert out == float(str(np.float16(0.1)))
    assert to_py_scalar(np.float64(1.0 / 3.0)) == 1.0 / 3.0
    assert np.isnan(to_py_scalar(np.float32("nan")))  # type: ignore[arg-type]
    assert to_py_scalar(np.float32("inf")) == float("inf")


def test_to_py_scalar_int_bool_passthrough() -> None:
    for value in (np.uint8(12), np.int64(-3), np.int32(7)):
        out = to_py_scalar(value)
        assert type(out) is int
        assert out == int(value)
    assert to_py_scalar(np.bool_(True)) is True
    assert to_py_scalar(np.bool_(False)) is False
    sentinel = object()
    assert to_py_scalar(sentinel) is sentinel
    assert to_py_scalar("x") == "x"
    assert to_py_scalar(1.5) == 1.5


# --------------------------------------------------------------------------- working directories


def test_create_working_directory_with_output_directory(tmp_path: Path) -> None:
    out = create_working_directory("snooping", tmp_path, "run1")
    assert out == tmp_path / "snooping" / "run1"
    assert out.is_dir()
    assert create_working_directory("snooping", str(tmp_path), "run1") == out  # exist_ok


def test_create_working_directory_timestamp_leaf(tmp_path: Path) -> None:
    out = create_working_directory("snooping", tmp_path)
    assert out.parent == tmp_path / "snooping"
    assert _TIMESTAMP_RE.match(out.name)
    assert out.is_dir()
    datetime.strptime(out.name, "%Y-%m-%d_%H-%M-%S")


def test_create_working_directory_empty_output_directory(tmp_path: Path) -> None:
    out = create_working_directory("snooping", tmp_path, "")
    assert out == tmp_path / "snooping"
    assert out.is_dir()
    assert list(out.iterdir()) == []


def test_create_working_directory_default_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = create_working_directory("snooping", None, "leaf")
    assert out == tmp_path / "working_directory" / "snooping" / "leaf"
    assert out.is_dir()
    ts = create_working_directory_with_timestamp("other")
    assert ts.parent == tmp_path / "working_directory" / "other"
    assert _TIMESTAMP_RE.match(ts.name)


def test_create_working_directory_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=f"Working directory {re.escape(str(missing))} does not exist."):
        create_working_directory("snooping", missing, "x")
    with pytest.raises(FileNotFoundError, match="does not exist."):
        create_working_directory("snooping", missing)
    with pytest.raises(FileNotFoundError, match="does not exist."):
        create_working_directory_with_timestamp("snooping", missing)
    assert not missing.exists()


# --------------------------------------------------------------------------- resolve_log_level


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("INFO", logging.INFO),
        ("info", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("WARN", logging.WARNING),
        ("warning", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        (logging.ERROR, logging.ERROR),
        (5, 5),
    ],
)
def test_resolve_log_level(level: str | int, expected: int) -> None:
    assert resolve_log_level(level) == expected


@pytest.mark.parametrize("level", ["NOPE", "TRACE", "", None, 1.5])
def test_resolve_log_level_invalid(level: object) -> None:
    with pytest.raises(ValueError):
        resolve_log_level(level)  # type: ignore[arg-type]


def test_resolve_log_level_verbose_warns_once(capsys: pytest.CaptureFixture[str]) -> None:
    assert resolve_log_level("VERBOSE") == logging.DEBUG
    assert capsys.readouterr().out == _VERBOSE_WARNING + "\n"
    assert resolve_log_level("verbose") == logging.DEBUG
    assert capsys.readouterr().out == ""
    adbg_log._VERBOSE_WARNED.clear()
    resolve_log_level("VERBOSE")
    assert capsys.readouterr().out == _VERBOSE_WARNING + "\n"


# --------------------------------------------------------------------------- get_logger


def test_get_logger_file_and_handlers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lg = get_logger("adbg_test_get_logger", tmp_path, level="DEBUG")
    out = capsys.readouterr().out
    assert out == f"File logs for adbg_test_get_logger will be stored at: {tmp_path}\n\n"
    assert lg is logging.getLogger("adbg_test_get_logger")
    assert lg.level == logging.DEBUG
    assert lg.propagate is False
    handlers = _marked(lg)
    assert len(handlers) == 2
    stream = [h for h in handlers if type(h) is logging.StreamHandler]
    files = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(stream) == 1
    assert len(files) == 1
    assert stream[0].level == logging.NOTSET
    rfh = files[0]
    assert Path(rfh.baseFilename) == tmp_path / f"accuracy_debugger_{LOG_FILE_SUFFIX}"
    assert rfh.maxBytes == 10485760
    assert rfh.backupCount == 40
    assert rfh.encoding == "utf8"
    for handler in handlers:
        assert handler.formatter is not None
        assert handler.formatter._fmt == LOG_FORMAT

    lg.info("hello file")
    rfh.flush()
    text = (tmp_path / "accuracy_debugger_user_info.log").read_text(encoding="utf8")
    assert re.search(r"^adbg_test_get_logger: .+ \| INFO \| hello file$", text, re.MULTILINE)


def test_get_logger_writes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lg = get_logger("adbg_test_stderr", tmp_path, "custom")
    capsys.readouterr()
    lg.warning("to stderr")
    captured = capsys.readouterr()
    assert "adbg_test_stderr: " in captured.err
    assert "| WARNING | to stderr" in captured.err
    assert "to stderr" not in captured.out
    assert (tmp_path / "custom_user_info.log").exists()


def test_get_logger_replaces_handlers_on_second_call(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    lg = get_logger("adbg_test_replace", first_dir)
    first_file = next(h for h in lg.handlers if isinstance(h, logging.FileHandler))
    foreign = logging.NullHandler()
    lg.addHandler(foreign)
    lg2 = get_logger("adbg_test_replace", second_dir, level="WARN")
    try:
        assert lg2 is lg
        assert lg.level == logging.WARNING
        assert len(_marked(lg)) == 2
        assert foreign in lg.handlers
        assert first_file not in lg.handlers
        assert first_file.stream is None  # closed
        lg.warning("second run")
        for h in lg.handlers:
            h.flush()
        assert "second run" in (second_dir / "accuracy_debugger_user_info.log").read_text(encoding="utf8")
        assert "second run" not in (first_dir / "accuracy_debugger_user_info.log").read_text(encoding="utf8")
    finally:
        lg.removeHandler(foreign)


def test_get_logger_verbose_maps_to_debug(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lg = get_logger("adbg_test_verbose", tmp_path, level="VERBOSE")
    assert lg.level == logging.DEBUG
    get_logger("adbg_test_verbose", tmp_path, level="VERBOSE")
    assert capsys.readouterr().out.count("'VERBOSE' is not a standard log level") == 1


def test_get_logger_creates_missing_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "a" / "b" / "c"
    get_logger("adbg_test_mkdir", str(log_dir))  # type: ignore[arg-type]
    assert log_dir.is_dir()
    assert (log_dir / "accuracy_debugger_user_info.log").exists()


def test_get_logger_is_picklable_by_name(tmp_path: Path) -> None:
    lg = get_logger("adbg_test_pickle", tmp_path)
    assert pickle.loads(pickle.dumps(lg)) is lg


# --------------------------------------------------------------------------- configure_worker_logging


def test_configure_worker_logging_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "worker.log"
    configure_worker_logging("adbg_test_worker", logging.INFO, log_file)
    configure_worker_logging("adbg_test_worker", logging.DEBUG, log_file)
    lg = logging.getLogger("adbg_test_worker")
    handlers = _marked(lg)
    assert len(handlers) == 2
    assert lg.propagate is False
    assert lg.level == logging.DEBUG
    assert sum(type(h) is logging.StreamHandler for h in handlers) == 1
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert not isinstance(file_handlers[0], logging.handlers.RotatingFileHandler)
    assert Path(file_handlers[0].baseFilename) == log_file
    assert file_handlers[0].mode == "a"
    assert not log_file.exists()  # delay=True

    lg.info("worker line")
    file_handlers[0].flush()
    text = log_file.read_text(encoding="utf8")
    assert re.fullmatch(r"adbg_test_worker: .+ \| INFO \| worker line\n", text)
    for handler in handlers:
        assert handler.formatter is not None
        assert handler.formatter._fmt == LOG_FORMAT


def test_configure_worker_logging_appends(tmp_path: Path) -> None:
    log_file = tmp_path / "worker.log"
    log_file.write_text("parent line\n", encoding="utf8")
    configure_worker_logging("adbg_test_worker_append", logging.INFO, log_file)
    lg = logging.getLogger("adbg_test_worker_append")
    lg.info("child line")
    for h in lg.handlers:
        h.flush()
    lines = log_file.read_text(encoding="utf8").splitlines()
    assert lines[0] == "parent line"
    assert lines[1].endswith("| INFO | child line")


def test_configure_worker_logging_without_file() -> None:
    configure_worker_logging("adbg_test_worker_nofile", logging.WARNING, None)
    lg = logging.getLogger("adbg_test_worker_nofile")
    handlers = _marked(lg)
    assert len(handlers) == 1
    assert type(handlers[0]) is logging.StreamHandler
    assert lg.level == logging.WARNING


def test_configure_worker_logging_replaces_get_logger_handlers(tmp_path: Path) -> None:
    get_logger("adbg_test_worker_mixed", tmp_path)
    configure_worker_logging("adbg_test_worker_mixed", logging.INFO, tmp_path / "w.log")
    lg = logging.getLogger("adbg_test_worker_mixed")
    assert len(_marked(lg)) == 2
    assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in lg.handlers)


# --------------------------------------------------------------------------- conftest fixtures


def test_reset_fixture_cleared_verbose_state() -> None:
    assert not adbg_log._VERBOSE_WARNED


def test_logger_fixtures(logger: logging.Logger, mock_logger: object) -> None:
    assert logger.name == "adbg_test"
    assert pickle.loads(pickle.dumps(logger)) is logger
    assert mock_logger.isEnabledFor(logging.DEBUG) is True  # type: ignore[attr-defined]


def test_chain_model_fixtures(qdq_chain_model: Path, fp32_chain_model: Path) -> None:
    import onnx  # noqa: PLC0415

    qdq = onnx.load(qdq_chain_model)
    assert [n.name for n in qdq.graph.node] == [
        "input_q",
        "input_dq",
        "w_dq",
        "conv1",
        "conv1_q",
        "conv1_dq",
        "relu1",
        "relu1_q",
        "relu1_dq",
    ]
    assert qdq.ir_version == 9
    assert qdq.opset_import[0].version == 19
    assert [i.name for i in qdq.graph.input] == ["input"]
    assert [o.name for o in qdq.graph.output] == ["output"]
    fp32 = onnx.load(fp32_chain_model)
    assert [n.name for n in fp32.graph.node] == ["conv1", "relu1"]
    assert [n.output[0] for n in fp32.graph.node] == ["conv1_out", "output"]


def test_chain_models_run_close(
    qdq_chain_model: Path, fp32_chain_model: Path, qdq_input_sample: dict[str, np.ndarray]
) -> None:
    ort = pytest.importorskip("onnxruntime")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    qdq_out = ort.InferenceSession(str(qdq_chain_model), so, providers=["CPUExecutionProvider"]).run(
        None, qdq_input_sample
    )[0]
    fp_out = ort.InferenceSession(str(fp32_chain_model), so, providers=["CPUExecutionProvider"]).run(
        None, qdq_input_sample
    )[0]
    assert qdq_out.shape == fp_out.shape == (1, 4, 8, 8)
    assert qdq_out.dtype == np.float32
    assert np.abs(qdq_out - fp_out).max() < 0.1


def test_write_input_list(tmp_path: Path, qdq_input_sample: dict[str, np.ndarray], write_input_list: object) -> None:
    path = write_input_list(qdq_input_sample, tmp_path / "inputs")  # type: ignore[operator]
    line = path.read_text(encoding="utf8").strip()
    name, raw = line.split(":=")
    raw = raw.strip('"')
    assert name == "input"
    assert Path(raw).is_absolute()
    data = np.fromfile(raw, dtype=np.float32).reshape(1, 3, 8, 8)
    np.testing.assert_array_equal(data, qdq_input_sample["input"])


def test_quantized_model_fixture(quantized_model: Path) -> None:
    import onnx  # noqa: PLC0415

    model = onnx.load(quantized_model)
    op_types = {n.op_type for n in model.graph.node}
    assert {"QuantizeLinear", "DequantizeLinear", "Conv", "Add"} <= op_types


def test_make_and_save_model(tmp_path: Path, make_model: object, save_model: object) -> None:
    from onnx import TensorProto, helper  # noqa: PLC0415

    model = make_model(  # type: ignore[operator]
        [helper.make_node("Relu", ["x"], ["y"], name="r")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        opset=13,
        ir_version=7,
    )
    assert model.ir_version == 7
    assert model.opset_import[0].version == 13
    path = save_model(model, tmp_path / "sub" / "m.onnx")  # type: ignore[operator]
    assert path.is_file()
