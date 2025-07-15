# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


import argparse
from datetime import datetime
from pathlib import Path

import qai_hub as hub
from pipeline.tasks.utils.checkpoint import Checkpoint


def main():
    parser = argparse.ArgumentParser(description="Compile a model to ONNX on Qualcomm AI Hub.")
    parser.add_argument("--model", required=True, help="Path to the model file to compile")
    args = parser.parse_args()

    model_path = args.model

    try:
        print(f"Compiling model from: {model_path}")
        checkpoint_path = Checkpoint.get_checkpoint_path_from_model_path(model_path)
        exported_model_id = Checkpoint.get_from_checkpoint(checkpoint_path, "exported_model_id")
        exported_model = hub.get_model(exported_model_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compiled_job = hub.submit_compile_job(
            exported_model,
            hub.Device("Snapdragon X Elite CRD"),
            name=f"{Path(model_path).name.removesuffix('.aimet')}_qairt_{timestamp}",
            options="--target_runtime onnx",
        )
        compiled_model = compiled_job.get_target_model()
        print(f"Model compiled successfully! Model ID: {compiled_model.model_id}")
        Checkpoint.add_to_checkpoint(checkpoint_path, "compiled_model_id", compiled_model.model_id)
    except Exception as e:
        print(f"Failed to compile model: {e}")


if __name__ == "__main__":
    main()
