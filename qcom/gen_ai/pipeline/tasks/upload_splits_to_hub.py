# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import subprocess

from ..task import Task
from .utils.checkpoint import Checkpoint


class UploadSplitsToHubTask(Task):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:
                checkpoint_path = Checkpoint.get_checkpoint_path_from_model_path(split.split_path)
                if not Checkpoint.has_checkpoint(checkpoint_path, "exported_model_id"):
                    return False

        return True

    def run_task(self) -> None:
        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:
                subprocess.run(
                    [self.model_dir.interpreter_path, "upload_to_hub.py", "--model", split.split_path], check=True
                )
                print(f"Uploaded split {split.split_name} to AI Hub...")
