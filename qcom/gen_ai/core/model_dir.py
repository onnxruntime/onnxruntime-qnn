# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import re

from pathlib import Path

class Split:

    def __init__(self, split_path):
        self.split_path = split_path
        self.split_name = Path(split_path).name
        self.checkpoint = {}

    @classmethod
    def from_checkpoint(self, ):




class ModelDir:

    def __init__(self, base_dir: str, model_name: str):
        self.model_name = model_name
        self.model_dir = os.path.join(base_dir, model_name)
        self.venv_dir = os.path.join(self.model_dir, "venv")
        self.interpreter_path = os.path.join(self.venv_dir, "bin", "python")
        # Dict[instantiation, Dict[split_name, Split]]
        self.splits = {}
        self.bootstrap()

    def _find_instantiations(self):
        return [d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, d)) and d != "venv"]

    def _find_splits(self):
        model_pattern = re.compile(r'(\d+)_of_\d+')
        instantiations = self._find_instantiations()
        logging.info(f"Found {len(instantiations)} instantiations: {instantiations}")
        for instantiation in instantiations:
            logging.debug(f"Found instantiation {instantiation}")
            checkpoint_files = [file for file in os.listdir(os.path.join(self.model_dir, instantiation)) if
                                file.endswith(".checkpoint")]
            sorted_splits = sorted(
                [Split.from_checkpoint(os.path.join(self.model_dir, instantiation, checkpoint)) for checkpoint
                 in checkpoint_files],
                key=lambda x: int(model_pattern.search(x.split_name).group(1))
            )

            logging.info(f"Found {len(sorted_splits)} checkpoints for instantiation {instantiation}")

            self.splits[instantiation] = sorted_splits

    def bootstrap(self):

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            logging.info(f"Creating model directory {self.model_dir}")
            os.makedirs(self.model_dir)
            return

        logging.info(f"Model directory {self.model_dir} already exists, attempting to find checkpoints...")

        # If it does exist, we should try to populate the ModelDir object
        self._find_splits()
        self._load_checkpoints_from_splits()
        self._validate_loaded_splits()


