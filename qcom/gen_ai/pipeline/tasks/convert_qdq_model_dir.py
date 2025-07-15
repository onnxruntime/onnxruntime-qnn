# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import subprocess

from ..task import Task


class ConvertQDQModelDirTask(Task):
    def __init__(self, model_dir, *, save_output=True):
        self.model_dir = model_dir
        self.save_output = save_output

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        return False

    def run_task(self) -> None:
        # Override the ORT in interpreter path with the local version
        self.model_dir.override_onnxruntime()

        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:
                if self.save_output:
                    logging.info(f"Converting {split.split_path} on ONNX and saving output...")
                    stdout_file = split.split_path.replace(".aimet", "_stdout.txt")
                    stderr_file = split.split_path.replace(".aimet", "_stderr.txt")
                    with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                        result = subprocess.run(
                            [self.model_dir.interpreter_path, "convert_qdq_to_onnx.py", "--model", split.split_path],
                            stdout=stdout,
                            stderr=stderr
                        )

                        if result.returncode != 0:
                            logging.error(f"Failed to fully convert model {split.split_name} to ONNX. Please check the logs for more details.")
                        else:
                            logging.info(f"Successfully converted split {split.split_name} to ONNX...")


