# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
import subprocess

from ..task import Task


class DownloadQDQModelDirTask(Task):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:
                onnx_qdq_path = os.path.join(
                    os.path.dirname(split.split_path), f"{split.split_path.replace('.aimet', '')}_qdq.onnx"
                )
                if not os.path.exists(onnx_qdq_path):
                    return False
        return True

    def run_task(self) -> None:
        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:
                subprocess.run(
                    [self.model_dir.interpreter_path, "download_qdq_from_hub.py", "--model", split.split_path],
                    check=True,
                )
                print(f"Downloaded split {split.split_name} from AI Hub...")
