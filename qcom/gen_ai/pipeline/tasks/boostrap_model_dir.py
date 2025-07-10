# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os

from ..task import Task


class BootstrapModelDirTask(Task):
    def __init__(self, model_dir, *, local_mode=False):
        self.model_dir = model_dir
        self.local_mode = local_mode

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        return self.model_dir.is_venv_bootstrapped()

    def run_task(self) -> None:
        self.model_dir.bootstrap_venv(local_mode=self.local_mode)
