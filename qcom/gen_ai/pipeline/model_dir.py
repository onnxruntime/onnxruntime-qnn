# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

ORT_QNN_WHEEL_PATH = os.path.join("..", "..", "build", "linux", "Release", "dist")


class Split:
    def __init__(self, split_path):
        self.split_path = split_path
        self.split_name = Path(split_path).name


class ModelDir:
    def __init__(self, base_dir: str, model_name: str):
        self.model_name = model_name
        self.model_dir = os.path.join(base_dir, model_name)
        self.venv_dir = os.path.join(self.model_dir, "venv")
        self.interpreter_path = os.path.join(self.venv_dir, "bin", "python")
        # Dict[instantiation, Dict[split_name, Split]]
        self.bootstrap()
        self.splits = self._find_splits()

    def _find_instantiations(self):
        return [d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, d)) and d != "venv"]

    def _find_splits(self):
        splits = {}
        split_pattern = re.compile(r".*(\d+)_of_\d+.aimet")
        instantiations = self._find_instantiations()
        logging.info(f"Found {len(instantiations)} instantiations: {instantiations}")
        for instantiation in instantiations:
            logging.debug(f"Found instantiation {instantiation}")
            sorted_splits = sorted(
                [
                    Split(os.path.join(self.model_dir, instantiation, split))
                    for split in os.listdir(os.path.join(self.model_dir, instantiation))
                    if split_pattern.match(split)
                ],
                key=lambda x: int(split_pattern.search(x.split_name).group(1)),
            )

            logging.info(f"Found {len(sorted_splits)} splits for instantiation {instantiation}")
            splits[instantiation] = sorted_splits
        return splits

    def bootstrap(self):
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            logging.info(f"Creating model directory {self.model_dir}")
            os.makedirs(self.model_dir)

    def is_venv_bootstrapped(self):
        return os.path.exists(os.path.join(self.model_dir, "venv.checkpoint"))

    def bootstrap_venv(self, *, local_mode=False):
        # Create the virtual environment using uv
        subprocess.run(["uv", "venv", "-p", "3.10", "--seed", self.venv_dir], check=True)
        logging.info(f"Created virtual environment in: {self.venv_dir}")

        # Install uv inside the virtual environment
        subprocess.run([self.interpreter_path, "-m", "pip", "install", "uv"], check=True)
        logging.info("Installed uv")

        # Install model-specific QAIHM package
        if local_mode:
            if "AI_HUB_MODELS_ROOT" not in os.environ:
                logging.error("AI_HUB_MODELS_ROOT environment variable not set, cannot install model-specific package")
                return
            qaihm_package = os.path.join(
                os.environ.get("AI_HUB_MODELS_ROOT"),
                "build",
                "wheel",
                "qai_hub_models-0.32.0-py3-none-any.whl",
            )
            qaihm_model_package = os.path.join(
                os.environ.get("AI_HUB_MODELS_ROOT"),
                "build",
                "wheel",
                f"qai_hub_models-0.32.0-py3-none-any.whl[{self.model_name}]",
            )
        else:
            qaihm_package = "qai_hub_models"
            qaihm_model_package = f"qai_hub_models[{self.model_name}]"

        # Prefer the model package, but if it doesn't exist, fall back to the base package
        try:
            subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", qaihm_model_package], check=True)
        except:
            subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", qaihm_package], check=True)
        print(f"Installed package: {qaihm_package}")

        subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", "qai-hub"], check=True)

        # Install model-specific dependencies?
        if "llama" in self.model_name:
            subprocess.run(
                [self.interpreter_path, "-m", "uv", "pip", "install", "-r", "llama-requirements.txt"], check=True
            )
            print("Installed packages from llama-requirements.txt")

        Path(os.path.join(self.model_dir, "venv.checkpoint")).touch()

    def override_onnxruntime(self):
        ort_wheel_candidates = os.listdir(ORT_QNN_WHEEL_PATH)

        print(ort_wheel_candidates)

        if len(ort_wheel_candidates) != 1:
            logging.error("No ONNX Runtime wheel found in the specified path")
            sys.exit(1)

        # Uninstall existing onnxruntime package to prevent any potential conflicts
        subprocess.run([self.interpreter_path, "-m", "uv", "pip", "uninstall", "onnxruntime"], check=True)

        ort_wheel = os.path.join(ORT_QNN_WHEEL_PATH, ort_wheel_candidates[0])
        subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", ort_wheel], check=True)

    def get_blank_results(self):
        results = {}
        for instantiation in self.splits:
            results[instantiation] = {}
            for split in self.splits[instantiation]:
                results[instantiation][split] = {}
        return results
