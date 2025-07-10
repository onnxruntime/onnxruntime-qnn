# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path

REPO_ROOT = (Path(__file__).parent / ".." / "..").resolve()


class TaskInfo:
    def __init__(
        self, description: str, dependencies: list[str], checkpoint_outputs: list[str], group_name: str | None = None
    ) -> None:
        """
        Initialize a new instance.

        Args:
          * :group_name: Used for logging and grouping messages. None is valid.
        """
        self.description = description
        self.dependencies = dependencies
        self.checkpoint_outputs = checkpoint_outputs
        self.group_name = group_name


class Task(ABC):
    def __init__(self) -> None:
        """
        Initialize a new instance.

        Args:
          * :group_name: Used for logging and grouping messages. None is valid.
        """

    @abstractmethod
    def does_work(self) -> bool:
        """
        Return True if this task actually does something (e.g., runs commands).
        """

    @abstractmethod
    def run_task(self) -> None:
        """
        Entry point for implementations: perform the task's action.
        """

    @abstractmethod
    def is_checkpointed(self) -> bool:
        """
        Return True if this task is checkpointed.
        """

    def run(self) -> None:
        """
        Entry point for callers: perform any startup/teardown tasks and call run_task.
        """
        self.run_task()
