# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

REPO_ROOT = (Path(__file__).parent / ".." / "..").resolve()


class Task:
    def __init__(self, description: str, dependencies: list[str], checkpoint_outputs: list[str], group_name: str | None = None) -> None:
        """
        Initialize a new instance.

        Args:
          * :group_name: Used for logging and grouping messages. None is valid.
        """
        self.description = description
        self.dependencies = dependencies
        self.checkpoint_outputs = checkpoint_outputs
        self.group_name = group_name
