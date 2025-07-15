# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
import logging
import os
import sys

from ..task import Task


class ConvertAIMETModelDirTask(Task):
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def does_work(self) -> bool:
        return True

    def is_checkpointed(self) -> bool:
        return False

    def run_task(self) -> None:
        if "QNN_SDK_ROOT" not in os.environ:
            logging.error("QNN_SDK_ROOT environment variable not set. Please set it to the root of the QAIRT SDK.")
            sys.exit(1)

        import qairt

        for instantiation in self.model_dir.splits:
            for split in self.model_dir.splits[instantiation]:

                # Identify the .onnx and .encodings file in the split directory
                onnx_network_path, encodings_path = None, None
                for file in os.listdir(split.split_path):
                    file_path = os.path.join(split.split_path, file)
                    if file_path.endswith(".onnx"):
                        onnx_network_path = file_path
                    if file_path.endswith(".encodings"):
                        encodings_path = file_path

                if not onnx_network_path or not encodings_path:
                    logging.error("ONNX and/or encodings file not found in split directory {split.split_path}.")
                    sys.exit(1)

                logging.info("Converting AIMET model on QAIRT to QNN model...")

                qairt_output_dir = split.split_path.replace(".aimet", "_qairt")
                os.makedirs(qairt_output_dir, exist_ok=True)

                qairt_output_path = os.path.join(qairt_output_dir, f"{split.split_name.replace('.aimet', '')}.dlc")
                qairt_converted_model = qairt.convert(onnx_network_path, encodings=encodings_path)
                logging.info(f"Saving QAIRT DLC to {qairt_output_path}")
                qairt_converted_model.save(qairt_output_path)




