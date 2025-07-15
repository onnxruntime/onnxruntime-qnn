# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


import argparse
import logging
import os
import tempfile
import zipfile

import qai_hub as hub
from pipeline.tasks.utils.checkpoint import Checkpoint


def main():
    parser = argparse.ArgumentParser(description="Download a QDQ model from Qualcomm AI Hub.")
    parser.add_argument("--model", required=True, help="Path to the model file to download")
    args = parser.parse_args()

    model_path = args.model

    try:
        print(f"Downloading QDQ model corresponding to: {model_path}")
        checkpoint_path = Checkpoint.get_checkpoint_path_from_model_path(model_path)
        qdq_model_id = Checkpoint.get_from_checkpoint(checkpoint_path, "compiled_model_id")
        qdq_model = hub.get_model(qdq_model_id)

        with tempfile.TemporaryDirectory() as tmpdirname:
            print(tmpdirname)
            print(model_path)
            temp_qdq_path = os.path.join(tmpdirname, f"{os.path.basename(model_path).replace('.aimet', '')}_qdq")
            downloaded_path = qdq_model.download(temp_qdq_path)
            print(f"Model downloaded successfully! Path: {downloaded_path}")

            final_qdq_path = os.path.join(os.path.dirname(model_path), f"{model_path.replace('.aimet', '')}_qdq.onnx")
            os.makedirs(final_qdq_path, exist_ok=True)

            print(f"Extracting zip file to {final_qdq_path}")
            with tempfile.TemporaryDirectory() as tmpzipcontents:
                # Extract all contents
                with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
                    zip_ref.extractall(tmpzipcontents)

                for file in os.listdir(tmpzipcontents):
                    if file == "model.onnx":
                        os.rename(os.path.join(tmpzipcontents, file), final_qdq_path)
                    else:
                        logging.error("Found unexpected file in zip: {file}")

            print(f"Extracted zip file to {final_qdq_path}")

    except Exception as e:
        print(f"Failed to download QDQ model: {e}")


if __name__ == "__main__":
    main()
