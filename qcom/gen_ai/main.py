# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from core.logging import initialize_logging
from core.model_dir import ModelDir
from core.plan import PUBLIC_TASKS, TASK_REGISTRY, Plan, public_task, task

MODELS_DIR = os.path.abspath("models")

VALID_MODELS = [
    "llama_v3_2_3b_instruct",
    "llama_v3_8b_instruct",
    "mistral_7b_instruct_v0_3",
    "phi_3_5_mini_instruct",
    "qwen2_7b_instruct",
    "stable_diffusion_v2_1"
]

ORT_QNN_PATH = os.path.join("..", "..", "build", "linux", "Release", "dist", "onnxruntime_qnn_qcom_internal-1.23.0-cp310-cp310-linux_x86_64.whl")


def hash_directory(directory_path, hash_algo='sha256'):
    print(f"Generating hash for {directory_path}")
    hash_func = hashlib.new(hash_algo)

    for root, dirs, files in sorted(os.walk(directory_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory_path)
            hash_func.update(relative_path.encode()) # Include file path in hash
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)

    print(f"Finished hash generation for {directory_path}")
    return hash_func.hexdigest()


class Split:

    def __init__(self, split_path):
        self.split_path = split_path
        self.split_name = Path(split_path).name
        self.sha = None
        self.hub_model = None
        self.qdq_model = None
        self.onnx_qdq_path = os.path.join(os.path.dirname(self.split_path), f"{self.split_name.replace('.aimet', '')}_qdq.onnx")
        self.compiled_model = None

    def _to_checkpoint(self):
        self.sha = hash_directory(self.split_path)
        lines = [self.sha, self.hub_model.model_id]
        if self.compiled_model:
            lines.append(self.compiled_model.model_id)
        with open(os.path.join(os.path.dirname(self.split_path), f"{self.split_name}.checkpoint"), "w") as f:
            f.writelines([line + "\n" for line in lines])

    def upload(self):
        self.hub_model = hub.upload_model(self.split_path)
        print(f"Uploaded split {self.split_name} to AI Hub...")
        self._to_checkpoint()

    def _read_data_from_checkpoint(self):
        with open(os.path.join(os.path.dirname(self.split_path), f"{self.split_name}.checkpoint")) as f:
            lines = f.readlines()
        self.sha = hash_directory(self.split_path)
        if self.sha != lines[0].strip():
            raise ValueError(f"Mismatch between stored SHA and SHA of file for {self.split_name}")
        self.hub_model = hub.get_model(lines[1].strip())

        if len(lines) >= 3:
            self.compiled_model = hub.get_model(lines[2].strip())

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        split = cls(checkpoint_path.replace(".checkpoint", ""))
        split._read_data_from_checkpoint()
        return split

    def compile_on_onnx(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Compiling {self.split_name} for ONNX using hub...")
        compile_job = hub.submit_compile_job(
            self.hub_model,
            hub.Device("Snapdragon X Elite CRD"),
            name=f"{self.split_name}_qairt_{timestamp}",
            options="--target_runtime onnx"
        )
        print(f"Submitted compile job {compile_job.job_id} for {self.split_name} on ONNX...")
        self.compiled_model = compile_job.get_target_model()
        self._to_checkpoint()

    def _get_qdq_model(self):
        print(f"Determining QDQ model ID for model {self.split_name}")
        compile_job = self.compiled_model.producer
        self.qdq_model = compile_job.get_target_model()

    def download_qdq_model(self):
        self._get_qdq_model()
        # Download the model to a temporary directory inside the model dir and then move it outside
        print(f"Downloading QDQ model to {self.onnx_qdq_path}")
        downloaded_path = self.qdq_model.download(self.onnx_qdq_path)
        print(f"Downloaded QDQ model to {downloaded_path}")

    def convert_w_ort_ep(self):
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.add_session_config_entry("ep.context_enable", "1")
        ep_context_onnx_model_path = os.path.join(os.path.dirname(self.split_path), f"{self.split_name.replace('.aimet', '')}_ep_converted.onnx")
        sess_options.add_session_config_entry("ep.context_file_path", ep_context_onnx_model_path)
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 1
        print(f"Saving EP context binary to {ep_context_onnx_model_path}")

        provider_options = [
            {'backend_path': 'libQnnHtp.so',
             'htp_graph_finalization_optimization_mode': 3,
             'soc_model': '60',
             'htp_arch': '73',
             'vtcm_mb': '8',
             'dump_json_qnn_graph': '1',
             'json_qnn_graph_dir': os.path.dirname(self.split_path)
             }
        ]

        sess = onnxruntime.InferenceSession(self.onnx_qdq_path,
                                            sess_options=sess_options,
                                            providers=['QNNExecutionProvider'],
                                            provider_options=provider_options)

        print("Successfully generated EP context binary.")

    def profile_on_qairt(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Profiling {self.split_name} for QAIRT using ORT...")
        profile_job = hub.submit_profile_job(
            self.compiled_model,
            hub.Device("Snapdragon X Elite CRD"),
            name=f"{self.split_name}_qairt_{timestamp}"
        )
        print(f"Submitted profile job {profile_job.job_id} for {self.split_name} on QAIRT...")


class ModelDirOld:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = os.path.abspath(self.model_name)
        self.venv_dir = os.path.join(model_name, "venv")
        self.interpreter_path = os.path.join(self.venv_dir, "bin", "python")

    def bootstrap(self, local_mode=False):
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Created directory: {self.model_dir}")

        # Create the virtual environment using uv
        subprocess.run(["uv", "venv", "-p", "3.10", "--seed", self.venv_dir], check=True)
        print(f"Created virtual environment in: {self.venv_dir}")

        # Install uv inside the virtual environment
        subprocess.run([self.interpreter_path, "-m", "pip", "install", "uv"], check=True)
        print("Installed uv")

        # Install model-specific QAIHM package
        if local_mode:
            qaihm_package = "../ai-hub-models-internal/build/release/wheel/qai_hub_models-0.30.6-py3-none-any.whl"
            qaihm_model_package = f"../ai-hub-models-internal/build/release/wheel/qai_hub_models-0.30.6-py3-none-any.whl[{self.model_name}]"
        else:
            qaihm_package = "qai_hub_models"
            qaihm_model_package = f"qai_hub_models[{self.model_name}]"

        # Prefer the model package, but if it doesn't exist, fall back to the base package
        try:
            subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", qaihm_model_package], check=True)
        except:
            subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", qaihm_package], check=True)
        print(f"Installed package: {qaihm_package}")

        # Install model-specific dependencies?
        if "llama" in self.model_name:
            subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", "-r", "llama-requirements.txt"],
                           check=True)
            print("Installed packages from llama-requirements.txt")

    def export(self):
        export_cmd = f"qai_hub_models.models.{self.model_name}.export"
        subprocess.run([self.interpreter_path, "-m", export_cmd, "--skip-compiling", "--output-dir", self.model_dir],
                       check=True)
        print(f"Successfully exported: {self.model_name}")



    def _find_splits_to_upload(self) -> dict[str, list[str]]:
        self.instantiations = self._find_instantiations()
        instantiations_contents = {d: [] for d in self.instantiations}

        model_pattern = re.compile(r'(\d+)_of_\d+\.aimet$')
        for instantiation in instantiations_contents:
            instantiation_models = os.listdir(os.path.join(self.model_dir, instantiation))
            sorted_splits = sorted([Split(os.path.join(self.model_dir, instantiation, model)) for model in instantiation_models if model_pattern.search(model)],
                                   key=lambda x: int(model_pattern.search(x.split_name).group(1))
                                   )
            instantiations_contents[instantiation] = sorted_splits

        return instantiations_contents

    def upload_splits(self):
        self.splits = self._find_splits_to_upload()
        for instantiation in self.splits:
            for split in self.splits[instantiation]:
                split.upload()

    def find_cached_splits(self) -> dict[str, list[str]]:
        self.instantiations = self._find_instantiations()
        instantiations_contents = {d: [] for d in self.instantiations}

        model_pattern = re.compile(r'(\d+)_of_\d+\.aimet$')
        for instantiation in instantiations_contents:
            checkpoint_files = [file for file in os.listdir(os.path.join(self.model_dir, instantiation)) if file.endswith(".checkpoint")]
            sorted_splits = sorted(
                [Split.from_checkpoint(os.path.join(self.model_dir, instantiation, checkpoint)) for checkpoint in checkpoint_files],
                key=lambda x: int(model_pattern.search(x.split_name).group(1))
                )
            instantiations_contents[instantiation] = sorted_splits

        self.splits = instantiations_contents

    def compile_on_onnx(self):
        for instantiation in self.splits:
            for split in self.splits[instantiation]:
                split.compile_on_onnx()

    def download_qdq_models(self):
        for instantiation in self.splits:
            for split in self.splits[instantiation]:
                split.download_qdq_model()

    def _override_onnxruntime_qnn(self):
        # Override the onnxruntime_qnn library with the one from the QDQ model
        print("Overriding onnxruntime_qnn library with the one from local build")
        subprocess.run([self.interpreter_path, "-m", "uv", "pip", "uninstall", "onnxruntime"], check=True)
        subprocess.run([self.interpreter_path, "-m", "uv", "pip", "install", ORT_QNN_PATH], check=True)
        print("Overridden onnxruntime_qnn library with the one from the QDQ model")

    def convert_w_ort_ep(self):
        self._override_onnxruntime_qnn()
        for instantiation in self.splits:
            for split in self.splits[instantiation]:
                split.convert_w_ort_ep()

    def profile_on_qairt(self):
        for instantiation in self.splits:
            for split in self.splits[instantiation]:
                split.profile_on_qairt()


@task(description="Bootstraps the model directory environment")
def bootstrap(plan: Plan):
    plan.model_dir.bootstrap()


@public_task(description="Exports a model from AI Hub Models to AI Hub", dependencies=["bootstrap"], checkpoint_outputs=["exported_model_id"])
def export(plan: Plan):
    plan.model_dir.export()


@public_task(description="Compiles an uploaded model to ONNX using AI Hub", dependencies=["export"], checkpoint_outputs=["compiled_onnx_model_id"])
def compile_on_onnx(plan: Plan):
    plan.model_dir.export()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates Gen AI models through a specified pipeline")
    parser.add_argument("--clean", action="store_true", help="Clean up the model directories")
    parser.add_argument("--model", help="Name of the model/package to be used")
    parser.add_argument(
        "task",
        type=str,
        help='Task to run. Specify "list" to show all tasks.',
    )
    parser.add_argument(
        "--force",
        type=str,
        action="append",
        help="Force a task to run even if it has already been completed"
    )

    return parser.parse_args()


def clean():
    models_dir = os.path.abspath("models")
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        subprocess.run(["rm", "-rf", models_dir], check=True)
        logging.info(f"Deleted directory: {models_dir}")
    else:
        logging.error(f"Models directory {models_dir} does not exist. Nothing to clean.")


def main():
    initialize_logging()
    args = parse_args()

    if os.path.abspath(os.path.dirname(__file__)) != os.path.abspath(os.getcwd()):
        logging.error("This script is only intended to run from the location of the script.")
        sys.exit(0)

    if args.clean:
        clean()
        sys.exit(0)

    model_name = args.model
    if model_name not in VALID_MODELS:
        logging.error(f"Invalid model name: {model_name}. Valid models are: {VALID_MODELS}")
        sys.exit(0)

    if args.task is None:
        logging.error("No task specified. Please specify a task to run.")
        sys.exit(0)

    if args.task == "list":
        logging.info("Available tasks:")
        for task_name in PUBLIC_TASKS:
            print(f"{task_name}: {TASK_REGISTRY[task_name].description}")
        sys.exit(0)

    if args.task not in TASK_REGISTRY:
        logging.error(f"Invalid task: {args.task}. Valid tasks are: {PUBLIC_TASKS}")
        sys.exit(0)

    # Create the models directory
    models_dir = os.path.abspath("models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    else:
        logging.warning(f"Models directory {models_dir} already exists, continuing...")

    model_dir = ModelDir(MODELS_DIR, model_name)

    plan = Plan(args.task, force=args.force or [])

    return

    model_dir = ModelDir(model_name)
    model_dir.bootstrap(local_mode=True)
    model_dir.find_cached_splits()

    if not args.skip_export:
        model_dir.export()

    if not args.skip_upload:
        model_dir.upload_splits()

    # Compile on QAIRT to get the QAIRT-compiled models
    if not args.skip_compile_on_onnx:
        model_dir.compile_on_onnx()

    if not args.skip_download_qdq_model:
        model_dir.download_qdq_models()

    if not args.skip_convert_w_ort_ep:
        model_dir.convert_w_ort_ep()

    # Profile on QAIRT
    if not args.skip_profile_on_qairt:
        model_dir.profile_on_qairt()


if __name__ == "__main__":
    main()
