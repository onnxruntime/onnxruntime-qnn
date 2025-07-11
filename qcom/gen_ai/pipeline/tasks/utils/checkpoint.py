# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os


class Checkpoint:
    @staticmethod
    def get_checkpoint_path_from_model_path(model_path: str):
        """
        Get the checkpoint path from the model path.
        """
        return model_path.removesuffix(".aimet") + ".checkpoint"

    @staticmethod
    def add_to_checkpoint(checkpoint_path: str, key: str, value: str):
        """
        Add a key-value pair to the checkpoint file.
        """
        with open(checkpoint_path, "a") as f:
            f.write(f"{key}={value}\n")

    @staticmethod
    def get_from_checkpoint(checkpoint_path: str, key: str) -> str:
        """
        Get the value of a key from the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            return None

        with open(checkpoint_path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split("=")[1].strip()
        return None

    @staticmethod
    def has_checkpoint(checkpoint_path: str, key: str) -> bool:
        """
        Check if a key exists in the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            return False

        with open(checkpoint_path) as f:
            for line in f:
                if line.startswith(key):
                    return True
        return False

    @staticmethod
    def remove_from_checkpoint(checkpoint_path: str, key: str):
        """
        Remove a key-value pair from the checkpoint file.
        """
        with open(checkpoint_path) as f:
            lines = f.readlines()
        with open(checkpoint_path, "w") as f:
            for line in lines:
                if not line.startswith(key):
                    f.write(line)

    @staticmethod
    def clear_checkpoint(checkpoint_path: str):
        """
        Clear the checkpoint file.
        """
        with open(checkpoint_path, "w") as f:
            f.write("")
