# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import subprocess
import sys

from pipeline.logging import initialize_logging
from pipeline.model_dir import ModelDir
from pipeline.plan import PUBLIC_TASKS, TASK_REGISTRY, Plan, public_task, task
from pipeline.tasks import (
    BootstrapModelDirTask,
    CompileModelDirTask,
    ConvertAIMETModelDirTask,
    ConvertQDQModelDirTask,
    DownloadQDQModelDirTask,
    ExportModelDirTask,
    UploadSplitsToHubTask,
)

MODELS_DIR = os.path.abspath("models")

VALID_MODELS = [
    "llama_v3_1_8b_instruct",
    "llama_v3_2_3b_instruct",
    "llama_v3_8b_instruct",
    "mistral_7b_instruct_v0_3",
    "phi_3_5_mini_instruct",
    "qwen2_7b_instruct",
    "stable_diffusion_v2_1",
]


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

    @public_task(
        description="Downloads the ONNX QDQ models from AI Hub",
        dependencies=["compile_on_onnx"],
    )
    def download_qdq(self, plan: Plan):
        return plan.add_step(DownloadQDQModelDirTask(self.model_dir))

    @public_task(description="Converts a downloaded QDQ model using ORT-QNN", dependencies=["download_qdq"])
    def convert_qdq_on_onnx(self, plan: Plan):
        return plan.add_step(ConvertQDQModelDirTask(self.model_dir))

    @public_task(description="Convert AIMET model using QAIRT", dependencies=["export"])
    def convert_aimet_on_qairt(self, plan: Plan):
        return plan.add_step(ConvertAIMETModelDirTask(self.model_dir))


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
