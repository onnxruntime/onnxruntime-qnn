// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// A JavaScript action's post step runs after all normal workflow steps, including
// when one fails or the job is cancelled. Preserve the workspace selected by the
// runner now because post steps must not infer a path from the current directory.
require("fs").appendFileSync(
  process.env.GITHUB_STATE,
  `workspace=${process.env.GITHUB_WORKSPACE || ""}\n` +
    `workspace_root=${process.env.RUNNER_WORKSPACE || ""}\n`,
);
