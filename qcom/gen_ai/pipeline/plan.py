# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import datetime
import logging
import sys
import time
from collections.abc import Callable

from .task import Task, TaskInfo

TASK_REGISTRY = {}
PUBLIC_TASKS = []


def task(description: str, dependencies: list[str] = None):
    def decorator(func):
        TASK_REGISTRY[func.__name__] = TaskInfo(description, dependencies or [])
        return func

    return decorator


def public_task(description: str, dependencies: list[str] = None):
    def decorator(func):
        PUBLIC_TASKS.append(func.__name__)
        TASK_REGISTRY[func.__name__] = TaskInfo(description, dependencies or [])
        return func

    return decorator


class Step:
    """A named Task within a Plan."""

    def __init__(self, step_id: str, task: Task):
        self._step_id = step_id
        self._task = task

    def __repr__(self) -> str:
        return self._step_id

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def task(self) -> Task:
        return self._task


class Plan:
    _steps: list[Step]
    _checkpoints: list[str]
    _forced: list[str]
    _plan_duration = datetime.timedelta | None
    _step_durations: list[tuple[str, datetime.timedelta]]

    def __init__(self, task: str, force=None):
        self._steps = []
        self._checkpoints = []
        self._forced = list(set(force or []))
        # mypy is wrong here:
        self._plan_duration = None  # type: ignore[assignment]
        self._step_durations = []

        self.task = TASK_REGISTRY[task]

    def add_step(self, task: Task, step_id: str | None = None) -> str:
        if step_id is None:
            # Default to the name of the calling function
            step_id = sys._getframe(1).f_code.co_name

        if self.count_step(step_id) > 10:
            raise RuntimeError(
                f"Refusing to add step '{step_id}' more than 10 times. Perhaps the planner is in an infinite loop?"
            )
        self._steps.append(Step(step_id, task))
        return step_id

    def force_step(self, step_id: str):
        if step_id not in self._forced:
            self._forced.append(step_id)

    def for_each(self, func: Callable[[str, Task], None]) -> None:
        for s in self._steps:
            func(s.step_id, s.task)

    def has_step(self, step_id: str) -> bool:
        return any(s.step_id == step_id for s in self._steps)

    def count_step(self, step_id: str) -> int:
        step_count = 0
        for s in self._steps:
            if s.step_id == step_id:
                step_count += 1
        return step_count

    def is_checkpointed(self, step_id: str) -> bool:
        step = [s for s in self._steps if s.step_id == step_id]
        if len(step) > 1:
            logging.error("Exception: Multiple steps with the same id: %s", step_id)
            sys.exit(1)
        return step[0].task.is_checkpointed()

    def print(self) -> None:
        for step in self._steps:
            step_msg = step.step_id
            if not step.task.does_work():
                step_msg += " (no-op)"
            if step.step_id in self._forced:
                step_msg += " (forced)"
            if self.is_checkpointed(step.step_id):
                step_msg += " (checkpointed)"
            print(step_msg)

    def run(self) -> None:
        start_time = time.monotonic()

        def run_task(step_id: str, task: Task) -> None:
            if self.is_checkpointed(step_id) and step_id not in self._forced:
                logging.warning(f"Checkpointed step {step_id}, skipping...")
            else:
                step_start_time = time.monotonic()

                caught: Exception | None = None
                try:
                    task.run()
                except Exception as ex:
                    logging.error(f"Error while running {step_id}.")
                    caught = ex
                step_end_time = time.monotonic()
                if task.does_work():
                    self._step_durations.append(
                        (
                            step_id,
                            datetime.timedelta(seconds=step_end_time - step_start_time),
                        )
                    )
                if caught is not None:
                    raise caught

        try:
            self.for_each(run_task)
        finally:
            end_time = time.monotonic()
            # mypy is wrong here:
            self._plan_duration = datetime.timedelta(seconds=end_time - start_time)  # type: ignore[assignment]
