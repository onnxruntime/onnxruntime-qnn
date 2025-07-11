# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pipeline.logging import initialize_logging
from pipeline.model_dir import ModelDir
from pipeline.plan import PUBLIC_TASKS, TASK_REGISTRY, Plan, public_task, task
from pipeline.tasks import BootstrapModelDirTask, CompileModelDirTask, ExportModelDirTask, UploadSplitsToHubTask

MODELS_DIR = os.path.abspath("models")

VALID_MODELS = [
    "llama_v3_2_3b_instruct",
    "llama_v3_8b_instruct",
    "mistral_7b_instruct_v0_3",
    "phi_3_5_mini_instruct",
    "qwen2_7b_instruct",
    "stable_diffusion_v2_1",
]

ORT_QNN_PATH = os.path.join(
    "..", "..", "build", "linux", "Release", "dist", "onnxruntime_qnn_qcom_internal-1.23.0-cp310-cp310-linux_x86_64.whl"
)


def hash_directory(directory_path, hash_algo="sha256"):
    print(f"Generating hash for {directory_path}")
    hash_func = hashlib.new(hash_algo)

    for root, dirs, files in sorted(os.walk(directory_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory_path)
            hash_func.update(relative_path.encode())  # Include file path in hash
            with open(file_path, "rb") as f:
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
        self.onnx_qdq_path = os.path.join(
            os.path.dirname(self.split_path), f"{self.split_name.replace('.aimet', '')}_qdq.onnx"
        )
        self.compiled_model = None

    def compile_on_onnx(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Compiling {self.split_name} for ONNX using hub...")
        compile_job = hub.submit_compile_job(
            self.hub_model,
            hub.Device("Snapdragon X Elite CRD"),
            name=f"{self.split_name}_qairt_{timestamp}",
            options="--target_runtime onnx",
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
        ep_context_onnx_model_path = os.path.join(
            os.path.dirname(self.split_path), f"{self.split_name.replace('.aimet', '')}_ep_converted.onnx"
        )
        sess_options.add_session_config_entry("ep.context_file_path", ep_context_onnx_model_path)
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 1
        print(f"Saving EP context binary to {ep_context_onnx_model_path}")

        provider_options = [
            {
                "backend_path": "libQnnHtp.so",
                "htp_graph_finalization_optimization_mode": 3,
                "soc_model": "60",
                "htp_arch": "73",
                "vtcm_mb": "8",
                "dump_json_qnn_graph": "1",
                "json_qnn_graph_dir": os.path.dirname(self.split_path),
            }
        ]

        sess = onnxruntime.InferenceSession(
            self.onnx_qdq_path,
            sess_options=sess_options,
            providers=["QNNExecutionProvider"],
            provider_options=provider_options,
        )

        print("Successfully generated EP context binary.")

    def profile_on_qairt(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Profiling {self.split_name} for QAIRT using ORT...")
        profile_job = hub.submit_profile_job(
            self.compiled_model, hub.Device("Snapdragon X Elite CRD"), name=f"{self.split_name}_qairt_{timestamp}"
        )
        print(f"Submitted profile job {profile_job.job_id} for {self.split_name} on QAIRT...")


class ModelDirOld:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = os.path.abspath(self.model_name)
        self.venv_dir = os.path.join(model_name, "venv")
        self.interpreter_path = os.path.join(self.venv_dir, "bin", "python")

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


class TaskLibrary:
    def __init__(self, model_dir: ModelDir):
        self.model_dir = model_dir

    @task(description="Bootstraps the model directory environment")
    def bootstrap(self, plan: Plan):
        return plan.add_step(BootstrapModelDirTask(self.model_dir, local_mode=True))

    @public_task(
        description="Exports a model from AI Hub Models to the model directory",
        dependencies=["bootstrap"],
    )
    def export(self, plan: Plan):
        return plan.add_step(ExportModelDirTask(self.model_dir))

    @public_task(
        description="Uploads the splits of a model to AI Hub",
        dependencies=["export"],
    )
    def upload_to_hub(self, plan: Plan):
        return plan.add_step(UploadSplitsToHubTask(self.model_dir))

    @public_task(
        description="Compiles an uploaded model to ONNX using AI Hub",
        dependencies=["upload_to_hub"],
    )
    def compile_on_onnx(self, plan: Plan):
        return plan.add_step(CompileModelDirTask(self.model_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates Gen AI models through a specified pipeline")
    parser.add_argument("--clean", action="store_true", help="Clean up the model directories")
    parser.add_argument("--dryrun", action="store_true", help="Dry run the pipeline")
    parser.add_argument("--model", help="Name of the model/package to be used")
    parser.add_argument(
        "task",
        nargs="*",
        type=str,
        help='Task to run. Specify "list" to show all tasks.',
    )
    parser.add_argument(
        "--force",
        type=str,
        action="append",
        help="Force a task and all affected tasks to run even if it has already been completed",
    )
    parser.add_argument(
        "--force_only", type=str, action="append", help="Force a task to run even if it has already been completed"
    )

    return parser.parse_args()


def clean():
    models_dir = os.path.abspath("models")
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        subprocess.run(["rm", "-rf", models_dir], check=True)
        logging.info(f"Deleted directory: {models_dir}")
    else:
        logging.error(f"Models directory {models_dir} does not exist. Nothing to clean.")


def plan_from_dependencies(model_dir: ModelDir, task: str, force: list[str], force_only: list[str]):
    plan = Plan(task)
    task_library = TaskLibrary(model_dir)

    work_list = [task]
    if not hasattr(task_library, task):
        logging.fatal(f"Task '{task}' does not exist.")
        sys.exit(1)

    while len(work_list) > 0:
        task_name = work_list.pop()
        if plan.has_step(task_name):
            continue
        unfulfilled_deps: list[str] = []
        for dep in TASK_REGISTRY.get(task_name).dependencies:
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                assert hasattr(task_library, dep), (
                    f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'."
                )

        if len(unfulfilled_deps) == 0:
            # add task_name to plan
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            added_step = task_adder(plan)
            assert added_step == task_name, (
                f"Task function '{task_name}' added a task with incorrect id '{added_step}'."
            )
        else:
            # Look at task_name again later when its deps are satisfied
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    # Add forced only tasks to plan
    for forced_task in force_only:
        plan.force_step(forced_task)

    # Add forced tasks based on forced list using depth-first search for UI-specified forced tasks
    for forced_task in force:
        if not plan.has_step(forced_task):
            logging.error(f"Forced task '{forced_task}' not found in plan")
            sys.exit(1)
        work_list.append(task)
        path = []
        while len(work_list) > 0:
            current_task = work_list.pop()
            path.append(current_task)

            # Return the path if we found our forced task
            if current_task == forced_task:
                break

            for dep in TASK_REGISTRY.get(current_task).dependencies:
                work_list.insert(0, dep)

        for task in path:
            plan.force_step(task)

    return plan


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

    if len(args.task) != 1:
        logging.error("Invalid number of tasks specified. Please specify exactly one task to run.")
        sys.exit(0)

    task = args.task[0]
    if task is None:
        logging.error("No task specified. Please specify a task to run.")
        sys.exit(0)

    if task == "list":
        logging.info("Available tasks:")
        for task_name in PUBLIC_TASKS:
            print(f"{task_name}: {TASK_REGISTRY[task_name].description}")
        sys.exit(0)

    if task not in TASK_REGISTRY:
        logging.error(f"Invalid task: {args.task}. Valid tasks are: {PUBLIC_TASKS}")
        sys.exit(0)

    # Create the models directory
    models_dir = os.path.abspath("models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    else:
        logging.warning(f"Models directory {models_dir} already exists, continuing...")

    model_dir = ModelDir(MODELS_DIR, model_name)
    plan = plan_from_dependencies(model_dir, task, force=args.force or [], force_only=args.force_only or [])

    if args.dryrun:
        plan.print()
        sys.exit(0)

    plan.run()


if __name__ == "__main__":
    main()
