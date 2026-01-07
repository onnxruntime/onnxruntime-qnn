#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, cast, get_args

# import jsonc
import numpy as np
import onnx
import yaml
from metric_factory import build_metrics_map

import onnxruntime

BackendT = Literal["cpu", "gpu", "htp"]


class ModelTestDef(NamedTuple):
    model_root: Path
    backend_type: BackendT
    enable_context: bool
    enable_cpu_fallback: bool
    metrics_config_path: Path = None

    def __repr__(self) -> str:
        return (
            f"{self.model_root.name}|{self.backend_type}"
            f"|enable_context:{self.enable_context}|cpu_fallback:{self.enable_cpu_fallback}"
        )


class ModelTestCase:
    def __init__(self, model_def: ModelTestDef) -> None:
        self.__model_root = model_def.model_root

        session_options = onnxruntime.SessionOptions()

        if not model_def.enable_cpu_fallback and model_def.backend_type != "cpu":
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        if model_def.enable_context:
            context_model_path = model_def.model_root / f"{model_def.model_root.name}_ctx.onnx"
            if context_model_path.exists():
                logging.debug(f"Clobbering stale context in {context_model_path}")
                context_model_path.unlink()
            session_options.add_session_config_entry("ep.context_enable", "1")
            session_options.add_session_config_entry("ep.context_file_path", str(context_model_path))

        logging.info(f"Preparing {self.__model_root.name}")
        self.__session = onnxruntime.InferenceSession(
            model_def.model_root / f"{model_def.model_root.name}.onnx",
            sess_options=session_options,
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_type": model_def.backend_type}],
        )

        # Parse metric config.
        metrics_config = self.__parse_yaml_config(model_def.metrics_config_path)
        # Build a mapping from output name -> Metric instance.
        self.__metrics_map = build_metrics_map(self.output_names, metrics_config)

    def __parse_yaml_config(self, config_path):
        """
        Load a YAML file into a Python dictionary with path validation.
        """
        if config_path:
            yaml_path = config_path
        else:
            yaml_path = self.__model_root / f"{self.__model_root.name}.yaml"

        # Check if the file exists and file type
        if not yaml_path.exists() or not yaml_path.is_file():
            logging.info(f"YAML file not found: {yaml_path}. Use default metric to evalute accuracy.")
            # Return empty dict to use default accuracy metric.
            return dict()

        # Load YAML
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {yaml_path}: {e}") from e

        # Validate structure
        if not isinstance(config, dict):
            raise ValueError(f"Expected a YAML mapping (dict) at top level, got: {type(config).__name__}")

        return config

    def load_inputs(self) -> list[dict[str, np.ndarray]]:
        return self.__tensors_from_files(self.input_paths)

    def load_outputs(self) -> list[dict[str, np.ndarray]]:
        return self.__tensors_from_files(self.output_paths)

    @property
    def input_names(self) -> list[str]:
        return [n.name for n in self.__session.get_inputs()]

    @property
    def input_shapes(self) -> list[list[int]]:
        return [n.shape for n in self.__session.get_inputs()]

    @property
    def input_paths(self) -> list[list[Path]]:
        return [list(ds.glob("input_*.pb")) for ds in self.__model_root.glob("test_data_set_*")]

    @property
    def output_names(self) -> list[str]:
        return [n.name for n in self.__session.get_outputs()]

    @property
    def output_shapes(self) -> list[list[int]]:
        return [n.shape for n in self.__session.get_outputs()]

    @property
    def output_paths(self) -> list[list[Path]]:
        return [list(ds.glob("output_*.pb")) for ds in self.__model_root.glob("test_data_set_*")]

    @staticmethod
    def __protos_from_files(data_sets: Iterable[Iterable[Path]]) -> list[list[onnx.TensorProto]]:
        return [[onnx.TensorProto.FromString(f.read_bytes()) for f in ds] for ds in data_sets]

    @classmethod
    def __tensors_from_files(cls, data_sets: Iterable[Iterable[Path]]) -> list[dict[str, np.ndarray]]:
        return [
            {node.name: onnx.numpy_helper.to_array(node) for node in dataset}
            for dataset in cls.__protos_from_files(data_sets)
        ]

    def run(self) -> None:
        inputs = self.load_inputs()
        expected = self.load_outputs()

        if len(inputs) == 0:
            logging.info(f"{self.__model_root.name} has no reference data.")
            return

        logging.info(f"Evaluating {self.__model_root.name} with {len(inputs)} reference datasets.")
        assert len(inputs) == len(expected)

        for ds_idx in range(len(inputs)):
            logging.debug(f"Inputs: { {n: t.shape for n, t in inputs[ds_idx].items()} }")
            logging.debug(f"Expected outputs: { {n: t.shape for n, t in expected[ds_idx].items()} }")
            actual = dict(
                zip(self.output_names, cast(Sequence[np.ndarray], self.__session.run([], inputs[ds_idx])), strict=False)
            )
            for name in expected[ds_idx]:
                if name not in actual:
                    logging.debug(f"Actual outputs: { {n: t.shape for n, t in actual.items()} }")
                    raise ValueError(f"Output {name} not found in actual.")

                if name in self.__metrics_map:
                    # Test with non-default metric
                    metric = self.__metrics_map[name]
                    logging.info(f"Comparing actual outputs for {name} to reference.")
                    val, passed = metric.evaluate(actual[name], expected[ds_idx][name])
                    if passed:
                        logging.info(f"{name} is close enough ({metric.name}: {val})")
                    else:
                        raise AssertionError(f"Accuracy check failed for output {name}. ({metric.name}: {val})")


class ModelTestSuite:
    def __init__(
        self,
        suite_root: Path,
        backend_type: BackendT,
        enable_context: bool,
        enable_cpu_fallback: bool,
    ) -> None:
        self.__suite_root = suite_root
        self.__backend_type: BackendT = backend_type
        self.__enable_context = enable_context
        self.__enable_cpu_fallback = enable_cpu_fallback

    @property
    def tests(self) -> Generator[ModelTestDef, None, None]:
        for test_root in self.__suite_root.iterdir():
            yield ModelTestDef(test_root, self.__backend_type, self.__enable_context, self.__enable_cpu_fallback)

    def run(self) -> None:
        for test in self.tests:
            ModelTestCase(test).run()


def initialize_logging(log_name: str) -> None:
    log_format = f"[%(asctime)s] [{log_name}] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", default="htp", choices=get_args(BackendT), help="QNN backend to use.")
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=None,
        help="Path to YAML file specifying metrics for outputs. Default metric is assert_allclose.",
    )
    parser.add_argument("--enable-context", action="store_true", help="[HTP only] Create a context cache.")
    parser.add_argument("--enable-cpu-fallback", action="store_true", help="Allow execution to fall back to CPU.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, metavar="MODEL_DIR", help="Path to a single model's directory.")
    model_group.add_argument("--suite", type=Path, metavar="SUITE_DIR", help="Path to a test suite directory.")

    args = parser.parse_args()

    initialize_logging("model_test.py")

    if args.model:
        ModelTestCase(
            ModelTestDef(
                args.model,
                args.backend,
                enable_context=args.enable_context,
                enable_cpu_fallback=args.enable_cpu_fallback,
                metrics_config_path=args.metrics_config,
            )
        ).run()
    elif args.suite:
        ModelTestSuite(
            args.suite,
            args.backend,
            enable_context=args.enable_context,
            enable_cpu_fallback=args.enable_cpu_fallback,
        ).run()
    else:
        raise RuntimeError("Unknown test mode")
