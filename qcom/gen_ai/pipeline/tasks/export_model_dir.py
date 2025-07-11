# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
import logging
import os
import re
import subprocess

from ..task import Task


class ExportModelDirTask(Task):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        if not os.listdir(self.model_dir.model_dir):
            return False

        # For each found instantiation, check that the highest number model is present
        expected_instantiations = ["prompt", "token"]
        detected_instantiations = [
            instantiation
            for instantiation in expected_instantiations
            if instantiation in os.listdir(self.model_dir.model_dir)
        ]
        if not detected_instantiations:
            return False

        for instantiation in detected_instantiations:
            instantiation_dir = os.path.join(self.model_dir.model_dir, instantiation)
            if not os.listdir(instantiation_dir):
                return False
            highest_split_pattern = r".*(\d)_of_\1.aimet"
            highest_split = [
                split for split in os.listdir(instantiation_dir) if re.fullmatch(highest_split_pattern, split)
            ]
            if len(highest_split) != 1:
                return False

        return True

    def run_task(self) -> None:
        logging.info(f"Exporting {self.model_dir.model_name}...")
        export_cmd = f"qai_hub_models.models.{self.model_dir.model_name}.export"
        subprocess.run(
            [
                self.model_dir.interpreter_path,
                "-m",
                export_cmd,
                "--skip-compiling",
                "--output-dir",
                self.model_dir.model_dir,
            ],
            check=True,
        )
        logging.info(f"Successfully exported: {self.model_dir.model_name}")
