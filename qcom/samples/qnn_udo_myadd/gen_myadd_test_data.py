# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Generate onnx_test_runner-format data for the MyAdd QDQ model.

The input matches the standalone samples: 32 float32 values evenly spaced in
[-1, 1]. The reference is the unquantized mathematical result (input +
constant); the on-device runner applies the documented QDQ tolerance.
"""

import argparse
from pathlib import Path

import numpy as np
from onnx import numpy_helper


def main():
    parser = argparse.ArgumentParser(description="Generate MyAdd ONNX test data")
    parser.add_argument("--constant", type=float, default=2.0, help="Value added by MyAdd (default: 2.0)")
    parser.add_argument("--outdir", required=True, help="Test-case directory that will contain test_data_set_0")
    args = parser.parse_args()

    data_dir = Path(args.outdir) / "test_data_set_0"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_data = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    output_data = input_data + np.float32(args.constant)

    with (data_dir / "input_0.pb").open("wb") as f:
        f.write(numpy_helper.from_array(input_data, name="input").SerializeToString())
    with (data_dir / "output_0.pb").open("wb") as f:
        f.write(numpy_helper.from_array(output_data, name="output").SerializeToString())

    print(f"Saved {data_dir}/input_0.pb and output_0.pb")


if __name__ == "__main__":
    main()
