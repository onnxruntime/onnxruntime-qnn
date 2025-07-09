# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


from .task import Task

TASK_REGISTRY = {}
PUBLIC_TASKS = []


def task(description: str, dependencies: list[str] = None, checkpoint_outputs: list[str] = None):
    def decorator(func):
        TASK_REGISTRY[func.__name__] = Task(description, dependencies or [], checkpoint_outputs or [])
        return func

    return decorator


def public_task(description: str, dependencies: list[str] = None, checkpoint_outputs: list[str] = None):
    def decorator(func):
        PUBLIC_TASKS.append(func.__name__)
        TASK_REGISTRY[func.__name__] = Task(description, dependencies or [], checkpoint_outputs or [])
        return func

    return decorator


class Plan:
    def __init__(self, task: str, model_name: str, force=None):
        self.task = TASK_REGISTRY[task]
        self.model_name = model_name
        self.force = set(force or [])
        self.tasks_to_run = []
        self.completed_tasks = set()
        self._determine_plan()

    def _determine_plan(self):

        # Determine what is in the checkpoints


        # Determine the list of tasks to run
        current_task = self.task


    def _print_debug(self):
        print(
            f"Task: {self.task}, force: {self.force}, completed_tasks: {self.completed_tasks}"
        )
