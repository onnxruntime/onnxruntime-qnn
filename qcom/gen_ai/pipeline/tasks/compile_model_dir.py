# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from ..task import Task


class CompileModelDirTask(Task):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        print("Compiling model directory")
