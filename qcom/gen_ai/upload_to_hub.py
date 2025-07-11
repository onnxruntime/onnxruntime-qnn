# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


import argparse

from pipeline.tasks.utils.checkpoint import Checkpoint
from qai_hub import upload_model


def main():
    parser = argparse.ArgumentParser(description="Upload a model to Qualcomm AI Hub.")
    parser.add_argument("--model", required=True, help="Path to the model file to upload")
    args = parser.parse_args()

    model_path = args.model

    try:
        print(f"Uploading model from: {model_path}")
        model = upload_model(model_path)
        print(f"Model uploaded successfully! Model ID: {model.model_id}")
        checkpoint_path = Checkpoint.get_checkpoint_path_from_model_path(model_path)
        Checkpoint.add_to_checkpoint(checkpoint_path, "exported_model_id", model.model_id)
    except Exception as e:
        print(f"Failed to upload model: {e}")


if __name__ == "__main__":
    main()
